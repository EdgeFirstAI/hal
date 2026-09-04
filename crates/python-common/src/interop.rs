// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Cross-package argument extraction.
//!
//! This is an `rlib`: every `edgefirst.*` extension module statically links
//! its own copy, so each ends up with its own PyO3 type objects.
//! `isinstance`/`downcast` between packages is therefore always false, even
//! though every module reports `__module__ == "edgefirst.tensor"` for the
//! type they all originate from. Entry points that used to require a
//! concrete `&PyTensor` accept `PyAny` instead and resolve it here:
//!
//! - **Same module** (the common case: a tensor created and consumed by the
//!   same extension, e.g. `ImageProcessor.create_image()` fed straight back
//!   into `convert()`): a direct runtime-borrow-checked reference into the
//!   caller's own `PyTensor` — no protocol round trip, no syscall, no copy.
//! - **Different module** (e.g. an `edgefirst.codec` tensor passed into
//!   `edgefirst.image`'s `ImageProcessor.convert()`): the
//!   `__edgefirst_tensor__` capsule protocol. The producer hands back a
//!   `PyCapsule` wrapping an `TensorDesc` (and, if a pin was requested, the
//!   `HostPin` that keeps its host address valid); we rebuild an *aliasing*
//!   `TensorDyn` from that descriptor via `TensorDyn::import_descriptor`.
//!
//! The capsule is the producer's pin's keepalive: dropping it early is a
//! use-after-free for anything that reads a `ptr` the descriptor carries, so
//! [`TensorArg`] retains it for as long as the imported tensor is in use.

use std::sync::Arc;

use edgefirst_tensor::{HostPin, TensorDesc, TensorDyn};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

/// Payload of the `edgefirst_tensor_v1` capsule -- the cross-package tensor
/// protocol's wire format.
///
/// `#[repr(C)]`: this crosses an `.so` boundary the same way
/// [`decoder_interop::DecoderCapsulePayload`] does, and for the identical
/// reason that struct is already `#[repr(C)]` -- a bare Rust tuple's field
/// order and padding are unspecified, so reading one back through a raw
/// pointer from a *different* compiled copy of this crate is not sound
/// even when today's compiler happens to lay it out the obvious way.
///
/// **Why `pin`/`pbo_keepalive` being `Arc<dyn Trait>` (a Rust-layout fat
/// pointer, not `#[repr(C)]`) does not contradict that.** Neither field is
/// ever read back by a *foreign* compiled copy of this crate. A consumer in
/// a different `.so` only ever reads `desc` (plain `#[repr(C)]` data) and
/// otherwise treats the whole capsule as opaque, holding the *Python
/// object* (`TensorArg::Foreign`'s `_keepalive: Py<PyAny>`) alive. The
/// keepalive fields' own `Drop` only ever runs when the capsule's Python
/// refcount reaches zero and PyCapsule invokes the destructor `PyCapsule::
/// new_with_value` registered -- generated code compiled into the SAME
/// `.so` that created the capsule (this producer's own copy of this crate),
/// so it drops `Self` -- including `pin`/`pbo_keepalive`'s vtables -- using
/// the exact layout that allocated it. Nothing here is ever interpreted by
/// a foreign compilation; only its *existence*, as an opaque blob a Python
/// refcount keeps alive, crosses the boundary.
///
/// Any change to this struct's layout -- including a change to *how* the
/// layout is guaranteed, not only to the fields themselves -- moves the
/// capsule name to `edgefirst_tensor_v2`, per INTEROP.md's Versioning rule.
#[repr(C)]
pub struct TensorCapsulePayload {
    pub desc: TensorDesc,
    pub pin: Option<HostPin<'static>>,
    /// Keepalive for a PBO-backed `desc`'s `ptr` (a `PboOpsVtable` address
    /// under `kind::PBO` -- see [`edgefirst_tensor::TensorDesc::ptr`]'s own
    /// doc comment). Mirrors `pin`'s role for the `HOST` kind exactly:
    /// without this, nothing keeps the producer's `PboHandle` (and the
    /// `OnceLock<PboOpsVtable>` field `ptr` addresses) alive between
    /// `__edgefirst_tensor__()` returning the capsule and a consumer
    /// eventually calling `import_descriptor` on it -- a genuine
    /// use-after-free if the producer's own Python tensor were garbage
    /// collected first. `None` for every kind but `PBO`.
    pub pbo_keepalive: Option<Arc<dyn Send + Sync>>,
}

/// A tensor argument resolved from an arbitrary Python object, valid for the
/// GIL lifetime `'py` of the `Bound` it was extracted from.
///
/// Obtained via [`TensorArg::extract`] (read-only, e.g. a `convert()`
/// source) or [`TensorArg::extract_mut`] (exclusive, e.g. a `convert()`
/// destination); read the access through the `AsRef`/`AsMut` impls below.
pub enum TensorArg<'py> {
    /// Same extension module: a live, runtime-borrow-checked reference into
    /// the caller's own `PyTensor`. Zero-copy *and* zero-syscall — nothing
    /// is cloned, duplicated or re-imported.
    NativeRef(PyRef<'py, crate::tensor::PyTensor>),
    /// Same extension module, exclusive access.
    NativeMut(PyRefMut<'py, crate::tensor::PyTensor>),
    /// A different extension module: imported via the capsule protocol.
    /// `_keepalive` is the producer's capsule — it owns the pin/vtable
    /// keepalive (if any) backing `tensor`'s `ptr`, and must outlive every
    /// use of `tensor`.
    ///
    /// Boxed: this variant carries a full owned `TensorDyn` while the
    /// native variants carry only a thin runtime-checked borrow, so an
    /// unboxed field would bloat every `TensorArg` (including the
    /// same-module fast path) to the foreign path's size.
    Foreign {
        tensor: Box<TensorDyn>,
        _keepalive: Py<PyAny>,
    },
}

impl<'py> TensorArg<'py> {
    /// Extract a read-only tensor argument, e.g. a `convert()` source.
    ///
    /// `access` is forwarded to `__edgefirst_tensor__` when the foreign-
    /// module path is taken; see that method's docs. Pass `None` on the
    /// GPU/DMA path (the common case — the descriptor's native handle is
    /// all a zero-copy consumer needs) and `Some("read")` when the caller
    /// genuinely needs the host address, e.g. reading model-output tensors
    /// on the CPU.
    ///
    /// A `None` request that comes back `HOST`-kind with no address (a
    /// Mem/Shm-backed producer — a JPEG decode is the common case, since
    /// software decoding writes to host memory) is retried once with
    /// `access="read"` rather than failing outright: those backends always
    /// pin successfully (see [`edgefirst_tensor::Tensor::pin_host`]'s
    /// cost docs), so there is nothing to lose by asking, and reading a
    /// decoded image cross-package is exactly the scenario this protocol
    /// exists for. `DMABUF`/`IOSURFACE`/`PBO`-kind descriptors never hit
    /// this arm — they carry a usable `handle` regardless of `ptr`, which
    /// is the case this crate must stay zero-cost for (see module docs).
    pub fn extract(obj: &Bound<'py, PyAny>, access: Option<&str>) -> PyResult<Self> {
        if let Ok(native) = obj.cast::<crate::tensor::PyTensor>() {
            return Ok(Self::NativeRef(native.try_borrow()?));
        }
        let retry_access = access.is_none().then_some("read");
        Self::from_protocol(obj, access, retry_access)
    }

    /// Extract an exclusive (write) tensor argument, e.g. a `convert()`
    /// destination or a `decode_into()` target.
    ///
    /// A heap-backed (`Mem`/`Shm`) destination is a legitimate target — the
    /// CPU fallback path exists precisely for tensors with no GPU backing —
    /// so this mirrors [`Self::extract`]'s retry: a `None` request that
    /// comes back `HOST`-kind with no address is retried once with
    /// `access="readwrite"` (not `"read"` — the destination is written, and
    /// a decode may read-modify-write into a strided destination too).
    /// `DMABUF`/`IOSURFACE`/`PBO`-kind descriptors never hit this arm — they
    /// carry a usable `handle` regardless of `ptr` — so the zero-pin fast
    /// path for GPU-importable destinations is unaffected. As with
    /// [`Self::extract`], an explicit `access` request is never retried:
    /// the caller already asked for exactly what it needs.
    pub fn extract_mut(obj: &Bound<'py, PyAny>, access: Option<&str>) -> PyResult<Self> {
        if let Ok(native) = obj.cast::<crate::tensor::PyTensor>() {
            return Ok(Self::NativeMut(native.try_borrow_mut()?));
        }
        let retry_access = access.is_none().then_some("readwrite");
        Self::from_protocol(obj, access, retry_access)
    }

    /// Whether [`Self::into_raw_access`] can succeed for `self`, i.e.
    /// whether it is safe to call `py.detach` around compute using this
    /// argument at all.
    ///
    /// `Foreign` is always detachable: it already owns an independent
    /// `TensorDyn`, built by exactly the reconstruction `into_raw_access`'s
    /// native path performs. `NativeRef`/`NativeMut` are detachable iff
    /// their backing memory is one `TensorDyn::import_descriptor` knows how
    /// to reconstruct -- `Mem`/`Shm` (host), `DmaBuf` (Linux), `IoSurface`
    /// (Apple) and, since `import_descriptor` grew a real `kind::PBO` arm
    /// (a `#[repr(C)]` vtable carried in the descriptor itself, plus
    /// `TensorCapsulePayload::pbo_keepalive` keeping the producer's
    /// `PboHandle` alive -- see that field's own doc comment), `Pbo` too.
    /// **Not** `Cuda`: `import_descriptor` has no arm for it (see its
    /// `k => Err(NotImplemented(...))` catch-all) -- unreachable today
    /// (`Tensor::new` rejects `Cuda`, and `TensorMemory::is_available`
    /// reports it unavailable), but excluded here so this predicate does
    /// not silently start lying the moment CUDA-backed tensors do land,
    /// rather than being discovered the same way `Pbo` was: by a live
    /// regression. A caller whose argument fails this check should keep the
    /// whole operation synchronous (GIL held throughout) rather than call
    /// `into_raw_access` -- exactly this crate's behaviour before GIL
    /// release existed for tensors at all, for this one remaining kind.
    pub fn can_detach(&self) -> bool {
        match self {
            Self::Foreign { .. } => true,
            Self::NativeRef(guard) => reconstructible(&guard.0),
            Self::NativeMut(guard) => reconstructible(&guard.0),
        }
    }

    /// The `__edgefirst_tensor__` capsule-protocol path, shared by
    /// [`Self::extract`] and [`Self::extract_mut`] once the fast
    /// same-module downcast has been ruled out.
    ///
    /// `retry_access`, when `Some`, is the access string to retry with if
    /// the first (`access`) call comes back `HOST`-kind with a null `ptr` —
    /// `"read"` for [`Self::extract`], `"readwrite"` for
    /// [`Self::extract_mut`]. The first call always requests no pin
    /// (`access` is `None` whenever a retry is armed), so there is no
    /// window where neither call holds a pin: nothing is dropped before the
    /// retry acquires one.
    fn from_protocol(
        obj: &Bound<'py, PyAny>,
        access: Option<&str>,
        retry_access: Option<&str>,
    ) -> PyResult<Self> {
        let method = obj.getattr("__edgefirst_tensor__").map_err(|_| {
            let type_name = obj
                .get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "<unknown>".to_string());
            pyo3::exceptions::PyTypeError::new_err(format!(
                "expected a tensor, got a {type_name}. Objects cross EdgeFirst packages via \
                 the __edgefirst_tensor__ capsule protocol; this object does not implement it. \
                 See https://github.com/EdgeFirstAI/hal/blob/main/crates/python-common/INTEROP.md \
                 for the protocol this method expects."
            ))
        })?;

        let (mut capsule_obj, mut desc) = Self::call_protocol(&method, access)?;
        if let Some(retry_access) = retry_access {
            if desc.kind == edgefirst_tensor::tensor_kind::HOST && desc.ptr.is_null() {
                (capsule_obj, desc) = Self::call_protocol(&method, Some(retry_access))?;
            }
        }

        let tensor = TensorDyn::import_descriptor(&desc).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "failed to import tensor via the __edgefirst_tensor__ protocol: {e}"
            ))
        })?;

        Ok(Self::Foreign {
            tensor: Box::new(tensor),
            _keepalive: capsule_obj.unbind(),
        })
    }

    /// Call `__edgefirst_tensor__(access=...)`, validate the returned
    /// capsule, and copy out its descriptor (`TensorDesc` is `Copy`, so
    /// this ends the unsafe borrow into the capsule's payload immediately
    /// rather than holding it across a possible retry that replaces
    /// `method`'s result).
    fn call_protocol(
        method: &Bound<'py, PyAny>,
        access: Option<&str>,
    ) -> PyResult<(Bound<'py, PyAny>, TensorDesc)> {
        let capsule_obj = match access {
            Some(a) => {
                let kwargs = pyo3::types::PyDict::new(method.py());
                kwargs.set_item("access", a)?;
                method.call((), Some(&kwargs))?
            }
            None => method.call0()?,
        };

        let capsule = capsule_obj.cast::<PyCapsule>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "__edgefirst_tensor__() must return a PyCapsule named \"edgefirst_tensor_v1\", \
                 per the cross-package tensor protocol",
            )
        })?;

        // `pointer_checked` validates the capsule name and returns the
        // payload pointer in one call -- no TOCTOU window between checking
        // the name and reading it. This name check is what makes the
        // unchecked-size read below sound: the name uniquely identifies
        // the payload's layout, and any future change to that layout moves
        // the name to `_v2` in the same commit (see INTEROP.md's
        // Versioning section). A producer built against a different name --
        // including a stale sibling `.so` in a partially-rebuilt
        // environment -- is rejected right here, before its payload is ever
        // read as this build's `TensorCapsulePayload` -- checking only
        // `desc.version` afterwards would be too late, since the
        // out-of-bounds/misaligned bytes are already read into `desc` by
        // the time `version` could be inspected.
        let ptr = capsule
            .pointer_checked(Some(c"edgefirst_tensor_v1"))
            .map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "__edgefirst_tensor__() returned a capsule not named \
                     \"edgefirst_tensor_v1\"; it does not follow the cross-package tensor \
                     protocol",
                )
            })?;
        // SAFETY: `pointer_checked` above confirmed the capsule is named
        // "edgefirst_tensor_v1", which by the protocol's contract (see
        // `PyTensor::__edgefirst_tensor__`) is only ever created via
        // `PyCapsule::new_with_value` wrapping exactly a
        // `TensorCapsulePayload`. `TensorDesc` is `Copy`; copying `.desc`
        // out ends the borrow here instead of tying it to the capsule's
        // lifetime.
        let desc = unsafe { (*(ptr.as_ptr() as *const TensorCapsulePayload)).desc };

        Ok((capsule_obj, desc))
    }
}

/// A resolved tensor argument with every Python guard already released, so
/// the holder may enter [`Python::detach`] (`py.detach`).
///
/// `TensorArg` itself cannot cross a detached region: its `NativeRef`/
/// `NativeMut` variants hold a `PyRef`/`PyRefMut`, and neither is `Ungil` --
/// precisely because each is a runtime-checked *borrow* of a Python object,
/// and `Ungil` exists to keep such borrows from crossing into a region where
/// no GIL is held to make "the object is still there, unmutated" true.
///
/// [`TensorArg::into_raw_access`] resolves that borrow away -- but dropping
/// the borrow flag is not by itself enough. An earlier version of this type
/// kept a raw pointer straight at the live `PyTensor`'s `TensorDyn` field,
/// reasoning that `TensorDyn` is unconditionally `Send + Sync` so nothing
/// stops it crossing into a detached region. That reasoning conflated two
/// different things: `Send`/`Sync` say a *type* may move between or be
/// shared across threads; they say nothing about two `&mut` to the same
/// *instance* coexisting, which is exactly what dropping the borrow flag
/// early allowed. Once the flag is clear, another Python thread -- the
/// thread this whole feature exists to unblock -- can call another
/// pymethod on the *same* `PyTensor`, take its own fresh `PyRefMut`, and
/// materialize a second live `&mut TensorDyn` aliasing the pointer a
/// detached closure is still dereferencing. That is undefined behaviour
/// under Rust's aliasing model (the compiler is entitled to assume `&mut`
/// exclusivity and optimize on it), not merely a documented data race.
///
/// The fix is to never alias the live value at all: [`TensorArg::NativeRef`]
/// / [`NativeMut`] reconstruct an **independent** `TensorDyn` via the same
/// [`TensorDesc`] + [`TensorDyn::import_descriptor`] machinery the
/// cross-package capsule protocol already uses (see the module docs) --
/// its own `shape_cache` etc., not a pointer into the original's. What this
/// does *not* remove: the reconstructed value still refers to the same
/// underlying buffer (dma-buf fd, host address, ...), so two threads
/// genuinely writing through two independent tensors backed by the same
/// memory can still race at the byte level. That residual is the same
/// caveat numpy's own C `nogil` sections carry -- "don't mutate the same
/// buffer from Python while a nogil section holds it" -- a documented
/// caller obligation, not a compiler-visible aliasing violation. `Foreign`
/// was never affected: it already owns a `Box<TensorDyn>` built exactly
/// this way, never a pointer into shared `PyClass` memory.
pub struct RawTensorAccess {
    repr: RawTensorRepr,
}

enum RawTensorRepr {
    /// Read-only origin: `TensorArg::NativeRef`, reconstructed here, or
    /// `TensorArg::Foreign` when it came from `TensorArg::extract`. Kept
    /// distinct from `Mut` so `AsMut::as_mut` can refuse to hand out a
    /// mutable view of something only ever extracted read-only -- same
    /// contract as `TensorArg::as_mut`. (`Foreign` itself draws no such
    /// distinction -- seeded into `Mut` below, matching
    /// `TensorArg::as_mut`'s existing behaviour of never panicking on it.)
    Ref {
        tensor: TensorDyn,
        _pin: Option<HostPin<'static>>,
        _keepalive: Option<Py<PyAny>>,
    },
    /// Exclusive origin: `TensorArg::NativeMut`, reconstructed here, or
    /// `TensorArg::Foreign` (either origin). Same shape as `Ref` otherwise.
    Mut {
        tensor: TensorDyn,
        _pin: Option<HostPin<'static>>,
        _keepalive: Option<Py<PyAny>>,
    },
}

// SAFETY: every field here is independently `Send`: `TensorDyn` is
// unconditionally `Send + Sync` (see its own safety comment) and, per this
// type's doc comment, is now always an independently-reconstructed value,
// never a pointer aliasing a still-borrowable Python object; `HostPin<
// 'static>` is `Send` (it already crosses the `PyCapsule` boundary bundled
// into `TensorCapsulePayload`, which requires `T: Send`); `Py<PyAny>` is
// `Send` unconditionally.
unsafe impl Send for RawTensorAccess {}

impl<'py> TensorArg<'py> {
    /// Resolve `self` into a [`RawTensorAccess`], dropping every Python
    /// guard so the caller may enter `py.detach` around the compute that
    /// follows.
    ///
    /// The native path reconstructs an independent `TensorDyn` (see
    /// [`RawTensorAccess`]'s docs), which -- like the capsule protocol's own
    /// import -- can fail: an unrecognised descriptor kind, or a pin the
    /// backend refuses. The foreign path never fails here: it was already
    /// holding an owned, `Send`, independently-built `TensorDyn`.
    pub fn into_raw_access(self) -> PyResult<RawTensorAccess> {
        use edgefirst_tensor::CpuAccess;

        Ok(match self {
            Self::NativeRef(guard) => {
                let (tensor, pin) = reconstruct(&guard.0, CpuAccess::Read)?;
                let py = guard.py();
                // `into_pyobject` cannot fail for a `PyRef`.
                let keepalive = guard.into_pyobject(py).unwrap().into_any().unbind();
                RawTensorAccess {
                    repr: RawTensorRepr::Ref {
                        tensor,
                        _pin: pin,
                        _keepalive: Some(keepalive),
                    },
                }
            }
            Self::NativeMut(guard) => {
                let (tensor, pin) = reconstruct(&guard.0, CpuAccess::ReadWrite)?;
                let py = guard.py();
                let keepalive = guard.into_pyobject(py).unwrap().into_any().unbind();
                RawTensorAccess {
                    repr: RawTensorRepr::Mut {
                        tensor,
                        _pin: pin,
                        _keepalive: Some(keepalive),
                    },
                }
            }
            Self::Foreign { tensor, _keepalive } => RawTensorAccess {
                repr: RawTensorRepr::Mut {
                    tensor: *tensor,
                    _pin: None,
                    _keepalive: Some(_keepalive),
                },
            },
        })
    }
}

/// Build an independent `TensorDyn` aliasing the same backing memory as
/// `tensor`, for [`TensorArg::into_raw_access`]'s native path. Requests no
/// pin unless the descriptor comes back `HOST`-kind with a null `ptr` (a
/// `Mem`/`Shm` producer -- same retry condition `TensorArg::extract`/
/// `extract_mut` already apply to the cross-package path), so the common
/// GPU/DMA case pays only a descriptor read plus whatever `import_descriptor`
/// itself costs (a `dup(2)` for `DMABUF`, a lookup for `IOSURFACE`) -- no
/// host pin forced onto a path that doesn't need one.
///
/// `pub(crate)`, not only reached through [`TensorArg::into_raw_access`]:
/// `tensor::PyTensor::decode_image`/`decode_image_file` call this directly
/// (see their doc comments) because, unlike a `RawTensorAccess` consumer,
/// they need to write the decode's resulting format/shape/colorimetry back
/// onto the *original* `PyTensor` afterward and so must keep their own
/// typed handle to it rather than letting `into_raw_access` fold it into an
/// opaque `Py<PyAny>` keepalive.
pub(crate) fn reconstruct(
    tensor: &TensorDyn,
    access: edgefirst_tensor::CpuAccess,
) -> PyResult<(TensorDyn, Option<HostPin<'static>>)> {
    let desc = tensor.descriptor_pinned(None);
    let (desc, pin) = if desc.kind == edgefirst_tensor::tensor_kind::HOST && desc.ptr.is_null() {
        let pin = tensor.pin_host(access).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "failed to resolve a tensor for a detached region: {e}"
            ))
        })?;
        let desc = tensor.descriptor_pinned(Some(&pin));
        (desc, Some(pin))
    } else {
        (desc, None)
    };
    let imported = TensorDyn::import_descriptor(&desc).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "failed to resolve a tensor for a detached region: {e}"
        ))
    })?;
    Ok((imported, pin))
}

/// Whether [`reconstruct`] can succeed for `tensor`'s backing memory --
/// shared by [`TensorArg::can_detach`] and, for `decoder`'s feature, each of
/// [`decoder_interop::ProtoDataArg`]'s two tensors independently. See
/// `TensorArg::can_detach`'s doc comment for why `Cuda` alone is excluded
/// now -- `Pbo` is not a capsule-crossing concern here at all:
/// [`reconstruct`] calls `import_descriptor` synchronously, in the same
/// function, on a `&TensorDyn` the borrow checker already keeps alive for
/// the whole call, so there is no window for the underlying `PboHandle` to
/// drop between building the descriptor and importing it -- unlike the
/// cross-package capsule path (`TensorArg::Foreign`), which needed its own
/// fix (`TensorCapsulePayload::pbo_keepalive`) for exactly that window.
pub(crate) fn reconstructible(tensor: &TensorDyn) -> bool {
    !matches!(tensor.memory(), edgefirst_tensor::TensorMemory::Cuda)
}

impl AsRef<TensorDyn> for RawTensorAccess {
    /// Borrow the resolved tensor. Same shape as [`TensorArg`]'s own
    /// `AsRef` impl.
    fn as_ref(&self) -> &TensorDyn {
        match &self.repr {
            RawTensorRepr::Ref { tensor, .. } => tensor,
            RawTensorRepr::Mut { tensor, .. } => tensor,
        }
    }
}

impl AsMut<TensorDyn> for RawTensorAccess {
    /// Mutably borrow the resolved tensor. Same shape as [`TensorArg`]'s own
    /// `AsMut` impl, including its panic contract.
    ///
    /// # Panics
    ///
    /// Panics if `self` was resolved from `TensorArg::extract` (read-only)
    /// rather than `extract_mut` -- an internal misuse, not a condition a
    /// caller can hit by passing an unexpected Python object.
    fn as_mut(&mut self) -> &mut TensorDyn {
        match &mut self.repr {
            RawTensorRepr::Mut { tensor, .. } => tensor,
            RawTensorRepr::Ref { .. } => {
                unreachable!("RawTensorAccess::as_mut called on a read-only extraction")
            }
        }
    }
}

impl<'py> AsRef<TensorDyn> for TensorArg<'py> {
    /// Borrow the resolved tensor.
    fn as_ref(&self) -> &TensorDyn {
        match self {
            Self::NativeRef(guard) => &guard.0,
            Self::NativeMut(guard) => &guard.0,
            Self::Foreign { tensor, .. } => tensor,
        }
    }
}

impl<'py> AsMut<TensorDyn> for TensorArg<'py> {
    /// Mutably borrow the resolved tensor.
    ///
    /// # Panics
    ///
    /// Panics if this `TensorArg` came from [`TensorArg::extract`] rather
    /// than [`TensorArg::extract_mut`] — an internal misuse, not a
    /// condition a caller can hit by passing an unexpected Python object.
    fn as_mut(&mut self) -> &mut TensorDyn {
        match self {
            Self::NativeMut(guard) => &mut guard.0,
            Self::Foreign { tensor, .. } => tensor,
            Self::NativeRef(_) => {
                unreachable!("TensorArg::as_mut called on a read-only extraction")
            }
        }
    }
}

// `Decoder` and `ProtoData` only exist when this crate's `decoder` domain is
// linked in (see Cargo.toml: `image` pulls in `decoder` too, since the fused
// `draw_masks`/`materialize_masks` paths are exactly what this module
// exists for).
#[cfg(feature = "decoder")]
mod decoder_interop {
    use super::{Bound, Py, PyAny, PyAnyMethods, PyCapsule, PyRef, PyResult, PyTypeMethods};
    use pyo3::types::PyCapsuleMethods;

    /// Payload of the `edgefirst_decoder_v1` capsule.
    ///
    /// `#[repr(C)]`: a bare Rust tuple's field order and padding are
    /// unspecified, so reading a field back at the wrong offset under
    /// toolchain drift would be undefined behaviour *before* the
    /// compatibility guard in [`DecoderArg::extract`] even runs. `repr(C)`
    /// fixes declaration-order fields and C padding rules instead.
    ///
    /// `decoder_size`/`decoder_align` are what the guard actually gates on
    /// (see `PyDecoder::__edgefirst_decoder__` for the full writeup). An
    /// earlier revision also carried a `version: &'static str` field as
    /// diagnostic-only text for the error message -- dropped, because a
    /// `&str` is itself a fat pointer whose two-word representation is not
    /// a layout the language guarantees either, the same category of
    /// hazard `#[repr(C)]` exists to close on the struct as a whole, and
    /// because the value it carried was `edgefirst-python-common`'s own
    /// `CARGO_PKG_VERSION` (this crate, where `PyDecoder` is defined), not
    /// `edgefirst-decoder`'s -- the error message named the wrong crate.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub(crate) struct DecoderCapsulePayload {
        pub(crate) ptr: usize,
        pub(crate) decoder_size: usize,
        pub(crate) decoder_align: usize,
    }

    /// A `Decoder` argument resolved from an arbitrary Python object, valid
    /// for the GIL lifetime `'py` of the `Bound` it was extracted from.
    ///
    /// Unlike [`super::TensorArg`], the foreign path here carries a raw
    /// pointer rather than an owned, reconstructed value — a `Decoder` is a
    /// live Rust object with internal post-processing state that cannot be
    /// decomposed the way a tensor can. See
    /// `crate::decoder::PyDecoder::__edgefirst_decoder__` for why that is
    /// sound only under a version guard, and for the hazard this is the one
    /// place in the protocol that depends on.
    pub enum DecoderArg<'py> {
        /// Same extension module: a live, runtime-borrow-checked reference
        /// into the caller's own `PyDecoder`.
        Native(PyRef<'py, crate::decoder::PyDecoder>),
        /// A different extension module: a borrowed pointer imported via
        /// the layout-guarded `__edgefirst_decoder__` capsule protocol.
        /// `_keepalive` is the producer object itself (the foreign
        /// `PyDecoder` instance `obj`, *not* the capsule -- the capsule's
        /// payload is plain data with no reference back to the Python
        /// object that produced it, so only retaining `obj` actually keeps
        /// the `Decoder` `decoder` points into alive). `decoder` is only
        /// guaranteed to outlive `_keepalive`, which is why this variant's
        /// lifetime is tied to the extraction's `'py`.
        ///
        /// Dereferencing `decoder` also bypasses PyO3's runtime borrow
        /// check on the producer's `PyDecoder` (the `RefCell`-like guard
        /// `Native`'s `PyRef` would give). That has no consequence while
        /// the caller keeps the GIL held for the whole call: no other
        /// Python thread can run concurrently, so nothing can race the
        /// read. It becomes a real aliasing hazard the instant a caller
        /// resolves a `Foreign` decoder and then releases the GIL
        /// (`py.detach`) while still holding this reference: a second
        /// Python thread could call a `&mut self` setter on the producer's
        /// live `PyDecoder` and materialise an aliasing `&mut Decoder`
        /// against it -- structurally the same UB `RawDecoderAccess` had
        /// before it was fixed (see `Decoder::decode`'s history).
        ///
        /// **Any caller that detaches around a resolved `DecoderArg` must
        /// gate on `matches!(arg, DecoderArg::Native(_))` first and keep
        /// the GIL held for `Foreign`** -- see
        /// `ImageProcessor::draw_masks`. There is no safe way to detach
        /// around a `Foreign` decoder today; nothing protects it the way
        /// `PyRef` protects `Native`, and making it safe would need real
        /// borrow protection added to the cross-package decoder capsule
        /// protocol itself.
        Foreign {
            decoder: &'py edgefirst_decoder::Decoder,
            _keepalive: Py<PyAny>,
        },
    }

    impl<'py> DecoderArg<'py> {
        /// Resolve a `Decoder` argument, e.g. `ImageProcessor.draw_masks`'s
        /// `decoder` parameter.
        pub fn extract(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
            if let Ok(native) = obj.cast::<crate::decoder::PyDecoder>() {
                return Ok(Self::Native(native.try_borrow()?));
            }

            let method = obj.getattr("__edgefirst_decoder__").map_err(|_| {
                let type_name = obj
                    .get_type()
                    .name()
                    .map(|n| n.to_string())
                    .unwrap_or_else(|_| "<unknown>".to_string());
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "expected a Decoder, got a {type_name}. Objects cross EdgeFirst packages \
                     via the __edgefirst_decoder__ capsule protocol; this object does not \
                     implement it."
                ))
            })?;
            let capsule_obj = method.call0()?;
            let capsule = capsule_obj.cast::<PyCapsule>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "__edgefirst_decoder__() must return a PyCapsule named \
                     \"edgefirst_decoder_v1\", per the cross-package decoder protocol",
                )
            })?;
            let ptr = capsule
                .pointer_checked(Some(c"edgefirst_decoder_v1"))
                .map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "__edgefirst_decoder__() returned a capsule not named \
                         \"edgefirst_decoder_v1\"; it does not follow the cross-package \
                         decoder protocol",
                    )
                })?;
            // SAFETY: `pointer_checked` confirmed the capsule is named
            // "edgefirst_decoder_v1", which by the protocol's contract (see
            // `PyDecoder::__edgefirst_decoder__`) is only ever created via
            // `PyCapsule::new_with_value` wrapping exactly a
            // `DecoderCapsulePayload`. The payload is `Copy`, so copying it
            // out ends the unsafe borrow here rather than tying it to the
            // capsule.
            let payload: DecoderCapsulePayload =
                unsafe { *(ptr.as_ptr() as *const DecoderCapsulePayload) };

            // The layout guard: see `PyDecoder::__edgefirst_decoder__` for
            // why matching `size_of`/`align_of` -- not a version string --
            // is what must hold before the pointer below can be
            // dereferenced at all. No package version is reported here: the
            // producer's own crate version isn't part of this payload (a
            // prior revision carried one, but it was `edgefirst-python-
            // common`'s version, not `edgefirst-decoder`'s, and mislabeled
            // as the latter), and `size`/`align` are what the guard
            // actually evaluated.
            let consumer_size = std::mem::size_of::<edgefirst_decoder::Decoder>();
            let consumer_align = std::mem::align_of::<edgefirst_decoder::Decoder>();
            if payload.decoder_size != consumer_size || payload.decoder_align != consumer_align {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Decoder crossed packages with a layout of size={}, align={}, but this \
                     package was built against a Decoder of size={consumer_size}, \
                     align={consumer_align}. These must describe the same Decoder layout -- \
                     reinstall matching versions of edgefirst-decoder and edgefirst-image (see \
                     crates/python-image/pyproject.toml's edgefirst-decoder pin).",
                    payload.decoder_size, payload.decoder_align,
                )));
            }

            // SAFETY: the layout guard above confirmed the producer and
            // consumer agree on `Decoder`'s size and alignment (a residual
            // risk remains: two layouts of equal size and alignment but
            // permuted fields would still pass -- see this variant's docs).
            // `_keepalive` retains `obj` itself, the producer `PyDecoder`
            // Python object -- not the capsule, whose payload holds no
            // reference back to it -- so `decoder` cannot outlive the
            // object it points into.
            let decoder = unsafe { &*(payload.ptr as *const edgefirst_decoder::Decoder) };
            Ok(Self::Foreign {
                decoder,
                _keepalive: obj.clone().unbind(),
            })
        }
    }

    impl<'py> AsRef<edgefirst_decoder::Decoder> for DecoderArg<'py> {
        fn as_ref(&self) -> &edgefirst_decoder::Decoder {
            match self {
                Self::Native(guard) => &guard.decoder,
                Self::Foreign { decoder, .. } => decoder,
            }
        }
    }
}

#[cfg(any(feature = "image", feature = "decoder"))]
mod proto_interop {
    use super::{
        Bound, HostPin, Py, PyAny, PyAnyMethods, PyCapsule, PyResult, PyTypeMethods,
        TensorCapsulePayload, TensorDyn,
    };
    // Only the `Native` variant below and its `into_raw_access` arm
    // (both feature = "decoder") name PyRef / call `into_pyobject`; an
    // "image"-only build (no "decoder") never reaches either.
    #[cfg(feature = "decoder")]
    use super::{IntoPyObject, PyRef};
    use edgefirst_tensor::ProtoData;
    use pyo3::types::PyCapsuleMethods;

    /// A `ProtoData` argument resolved from an arbitrary Python object.
    ///
    /// Unlike [`DecoderArg`], this carries no raw pointer and needs no
    /// version guard: `ProtoData` is just two tensors and an enum (see
    /// `edgefirst_tensor::ProtoData`), so the foreign path composes the
    /// existing, already-proven `__edgefirst_tensor__` protocol instead of
    /// describing `ProtoData`'s own layout. See
    /// `crate::decoder::PyProtoData::__edgefirst_protodata__`.
    pub enum ProtoDataArg<'py> {
        /// Same extension module.
        #[cfg(feature = "decoder")]
        Native(PyRef<'py, crate::decoder::PyProtoData>),
        /// A different extension module: an owned `ProtoData` reconstructed
        /// from two `__edgefirst_tensor__` capsules plus a layout code.
        /// Owned (not borrowed) because it is built fresh from imported
        /// tensor descriptors, just like `TensorArg::Foreign`.
        ///
        /// Boxed for the same reason as `TensorArg::Foreign`: `ProtoData`
        /// carries two full `TensorDyn`s, so an unboxed field would bloat
        /// every `ProtoDataArg` (including the same-module fast path) to
        /// the foreign path's size.
        ///
        /// `_keepalive` retains *both* producer capsules -- exactly as
        /// `TensorArg::Foreign` retains its single one. Each capsule owns a
        /// `TensorCapsulePayload`: dropping it releases the pin,
        /// and for a `Mem`-backed producer tensor that happens to be
        /// harmless (the allocation is a shared `Arc`), but a `Shm`-backed
        /// one would `munmap` out from under `proto_data`'s still-live
        /// `TensorDyn`s. Retaining both capsules for as long as
        /// `proto_data` is in scope is what makes that safe regardless of
        /// backend.
        Foreign {
            proto_data: Box<ProtoData>,
            _keepalive: (Py<PyAny>, Py<PyAny>),
            _lifetime: std::marker::PhantomData<&'py ()>,
        },
    }

    impl<'py> ProtoDataArg<'py> {
        /// Resolve a `ProtoData` argument, e.g.
        /// `ImageProcessor.materialize_masks`'s `proto_data` parameter.
        pub fn extract(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
            #[cfg(feature = "decoder")]
            if let Ok(native) = obj.cast::<crate::decoder::PyProtoData>() {
                return Ok(Self::Native(native.try_borrow()?));
            }

            let method = obj.getattr("__edgefirst_protodata__").map_err(|_| {
                let type_name = obj
                    .get_type()
                    .name()
                    .map(|n| n.to_string())
                    .unwrap_or_else(|_| "<unknown>".to_string());
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "expected a ProtoData, got a {type_name}. Objects cross EdgeFirst packages \
                     via the __edgefirst_protodata__ capsule protocol; this object does not \
                     implement it."
                ))
            })?;
            let result = method.call0()?;
            let (mask_cap, protos_cap, layout): (Bound<'py, PyAny>, Bound<'py, PyAny>, String) =
                result.extract().map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "__edgefirst_protodata__() must return (mask_coefficients_capsule, \
                         protos_capsule, layout_str), per the cross-package ProtoData protocol",
                    )
                })?;

            let mask_coefficients = import_tensor_capsule(&mask_cap)?;
            let protos = import_tensor_capsule(&protos_cap)?;
            let layout = match layout.as_str() {
                "nhwc" => edgefirst_tensor::ProtoLayout::Nhwc,
                "nchw" => edgefirst_tensor::ProtoLayout::Nchw,
                other => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "__edgefirst_protodata__() returned unknown layout {other:?}, expected \
                         \"nhwc\" or \"nchw\""
                    )));
                }
            };

            Ok(Self::Foreign {
                proto_data: Box::new(ProtoData {
                    mask_coefficients,
                    protos,
                    layout,
                }),
                _keepalive: (mask_cap.unbind(), protos_cap.unbind()),
                _lifetime: std::marker::PhantomData,
            })
        }
    }

    impl<'py> AsRef<ProtoData> for ProtoDataArg<'py> {
        fn as_ref(&self) -> &ProtoData {
            match self {
                #[cfg(feature = "decoder")]
                Self::Native(guard) => &guard.0,
                Self::Foreign { proto_data, .. } => proto_data,
            }
        }
    }

    /// A `ProtoData` resolved for `py.detach`, mirroring [`super::
    /// TensorArg::into_raw_access`]: `ProtoData` is just two tensors and an
    /// enum (see its own doc comment), so -- like a tensor, and unlike
    /// `Decoder`/`PyByteTrack` -- it is reconstructed into an independent
    /// value rather than borrowed through a held guard. That distinction
    /// matters here in the caller's favour: `proto_data` is produced once by
    /// `Decoder.decode_proto` and consumed once by `materialize_masks`/
    /// `draw_masks`, never observed for live updates the way a decoder's
    /// thresholds are, so reconstruction carries none of the "silently
    /// snapshots config the caller expects to keep observing" hazard that
    /// rules reconstruction out for a genuinely live mutable object -- see
    /// the H4b brief's framing, echoed in `RawTensorAccess`'s own doc
    /// comment.
    pub struct RawProtoDataAccess {
        proto_data: ProtoData,
        _pins: (Option<HostPin<'static>>, Option<HostPin<'static>>),
        _keepalive: RawProtoKeepalive,
    }

    enum RawProtoKeepalive {
        #[cfg(feature = "decoder")]
        Native { _guard: Py<PyAny> },
        Foreign {
            _mask: Py<PyAny>,
            _protos: Py<PyAny>,
        },
    }

    // SAFETY: see `RawTensorAccess` -- same argument, applied to two
    // independently-reconstructed `TensorDyn`s (via `super::reconstruct`)
    // instead of one; `RawProtoKeepalive`'s `Py<PyAny>`s are `Send`
    // unconditionally.
    unsafe impl Send for RawProtoDataAccess {}

    impl<'py> ProtoDataArg<'py> {
        /// Whether [`Self::into_raw_access`] can succeed for `self`. See
        /// [`super::TensorArg::can_detach`]'s doc comment for the underlying
        /// `import_descriptor` gap (`Pbo`/`Cuda`); it applies to each of
        /// `mask_coefficients`/`protos` independently, so both tensors must
        /// be detachable for the pair to be.
        pub fn can_detach(&self) -> bool {
            match self {
                #[cfg(feature = "decoder")]
                Self::Native(guard) => {
                    super::reconstructible(&guard.0.mask_coefficients)
                        && super::reconstructible(&guard.0.protos)
                }
                Self::Foreign { .. } => true,
            }
        }

        /// Resolve `self` into a [`RawProtoDataAccess`], dropping every
        /// Python guard so the caller may enter `py.detach`. See
        /// [`Self::can_detach`]: callers should check that first and take
        /// the synchronous path otherwise, the same way `TensorArg`
        /// consumers do.
        pub fn into_raw_access(self) -> PyResult<RawProtoDataAccess> {
            #[cfg(feature = "decoder")]
            use edgefirst_tensor::CpuAccess;

            match self {
                #[cfg(feature = "decoder")]
                Self::Native(guard) => {
                    // Both tensors are read-only here: `materialize_masks`/
                    // `draw_masks` only ever read `proto_data`.
                    let (mask_coefficients, mask_pin) =
                        super::reconstruct(&guard.0.mask_coefficients, CpuAccess::Read)?;
                    let (protos, protos_pin) =
                        super::reconstruct(&guard.0.protos, CpuAccess::Read)?;
                    let layout = guard.0.layout;
                    let py = guard.py();
                    // `into_pyobject` cannot fail for a `PyRef`.
                    let keepalive = guard.into_pyobject(py).unwrap().into_any().unbind();
                    Ok(RawProtoDataAccess {
                        proto_data: ProtoData {
                            mask_coefficients,
                            protos,
                            layout,
                        },
                        _pins: (mask_pin, protos_pin),
                        _keepalive: RawProtoKeepalive::Native { _guard: keepalive },
                    })
                }
                Self::Foreign {
                    proto_data,
                    _keepalive: (mask_keepalive, protos_keepalive),
                    ..
                } => Ok(RawProtoDataAccess {
                    proto_data: *proto_data,
                    _pins: (None, None),
                    _keepalive: RawProtoKeepalive::Foreign {
                        _mask: mask_keepalive,
                        _protos: protos_keepalive,
                    },
                }),
            }
        }
    }

    impl AsRef<ProtoData> for RawProtoDataAccess {
        /// Borrow the resolved proto data. Same shape as [`ProtoDataArg`]'s
        /// own `AsRef` impl.
        fn as_ref(&self) -> &ProtoData {
            &self.proto_data
        }
    }

    /// Validate and import a single `edgefirst_tensor_v1` capsule -- the
    /// same capsule shape `TensorArg` imports, factored out here because
    /// `__edgefirst_protodata__` returns two of them directly rather than a
    /// producer method to call.
    fn import_tensor_capsule(capsule_obj: &Bound<'_, PyAny>) -> PyResult<TensorDyn> {
        let capsule = capsule_obj.cast::<PyCapsule>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "__edgefirst_protodata__() must return PyCapsule objects named \
                 \"edgefirst_tensor_v1\", per the cross-package tensor protocol",
            )
        })?;
        let ptr = capsule
            .pointer_checked(Some(c"edgefirst_tensor_v1"))
            .map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "__edgefirst_protodata__() returned a capsule not named \
                     \"edgefirst_tensor_v1\"; it does not follow the cross-package tensor \
                     protocol",
                )
            })?;
        // SAFETY: see `TensorArg::call_protocol` -- same capsule shape,
        // same contract.
        let desc = unsafe { (*(ptr.as_ptr() as *const TensorCapsulePayload)).desc };
        TensorDyn::import_descriptor(&desc).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "failed to import a ProtoData tensor via the __edgefirst_tensor__ protocol: {e}"
            ))
        })
    }
}
#[cfg(feature = "decoder")]
pub use decoder_interop::DecoderArg;
#[cfg(feature = "decoder")]
pub(crate) use decoder_interop::DecoderCapsulePayload;
#[cfg(any(feature = "image", feature = "decoder"))]
pub use proto_interop::{ProtoDataArg, RawProtoDataAccess};
