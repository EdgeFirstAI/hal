// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Cross-package argument extraction.
//!
//! Every `edgefirst.*` extension module statically links its own copy of this
//! crate and has its own PyO3 type objects. Entry points accept `PyAny` and
//! resolve it here:
//!
//! - Same module: a runtime-borrow-checked reference into the caller's own
//!   `PyTensor`.
//! - Different module: the `__edgefirst_tensor__` capsule protocol. The
//!   producer returns a `PyCapsule` wrapping a `TensorDesc` and, if a pin was
//!   requested, the `HostPin` that keeps its host address valid; an aliasing
//!   `TensorDyn` is rebuilt from the descriptor via
//!   `TensorDyn::import_descriptor`.
//!
//! [`TensorArg`] retains the capsule, which keeps the producer's pin alive,
//! for as long as the imported tensor is in use.

use std::sync::Arc;

use edgefirst_tensor::{HostPin, Quantization, TensorDesc, TensorDyn};

/// The capsule's quantization descriptor, re-exported from `edgefirst-tensor`'s
/// `protocol` module.
pub use edgefirst_tensor::protocol::QuantDesc;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

/// Attach `quant` to `tensor`, if there is any. Returns an error when the
/// quantization does not validate against the tensor.
fn apply_quantization(tensor: &mut TensorDyn, quant: Option<Quantization>) -> PyResult<()> {
    if let Some(q) = quant {
        tensor.set_quantization(q).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "tensor quantization does not match the imported descriptor: {e}"
            ))
        })?;
    }
    Ok(())
}

/// The metadata a consumer copies out of a capsule before it rebuilds the
/// tensor.
pub struct CapsuleParts {
    pub desc: TensorDesc,
    pub quant: Option<Quantization>,
    pub plane_offset: u64,
}

/// Copy [`CapsuleParts`] out of a validated capsule payload.
///
/// # Safety
///
/// `ptr` must address the payload of a live `PyCapsule` whose name
/// [`PyCapsule::pointer_checked`] has confirmed to be `edgefirst_tensor_v2`.
/// The capsule must stay alive across the call.
unsafe fn read_capsule_parts(ptr: *const TensorCapsulePayload) -> PyResult<CapsuleParts> {
    // Each field is read by projection through the raw pointer; only the
    // `#[repr(C)]` prefix of the payload is read.
    //
    // SAFETY: `ptr` addresses a live `edgefirst_tensor_v2` payload, so its
    // `#[repr(C)]` prefix is valid.
    let desc = unsafe { (*ptr).desc };
    // SAFETY: as above.
    let quant_desc = unsafe { (*ptr).quant };
    // SAFETY: as above.
    let plane_offset = unsafe { (*ptr).plane_offset };
    Ok(CapsuleParts {
        desc,
        // SAFETY: the capsule that owns the borrowed scale arrays is alive
        // while `to_quantization` copies them out.
        quant: unsafe { quant_desc.to_quantization() }
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
        plane_offset,
    })
}

/// Compile-time checks of the layout of the wire format's `#[repr(C)]` prefix.
const _: () = {
    use std::mem::{align_of, offset_of, size_of};
    assert!(size_of::<QuantDesc>() == 2 * size_of::<u64>() + 2 * size_of::<*const ()>());
    assert!(align_of::<QuantDesc>() == align_of::<u64>());
    assert!(offset_of!(QuantDesc, len) == 0);
    assert!(offset_of!(QuantDesc, axis_plus_one) == size_of::<u64>());
    assert!(offset_of!(QuantDesc, scales) == 2 * size_of::<u64>());
    assert!(offset_of!(QuantDesc, zero_points) == 2 * size_of::<u64>() + size_of::<*const ()>());
    assert!(offset_of!(TensorCapsulePayload, desc) == 0);
    assert!(offset_of!(TensorCapsulePayload, quant) == size_of::<TensorDesc>());
    assert!(
        offset_of!(TensorCapsulePayload, plane_offset)
            == size_of::<TensorDesc>() + size_of::<QuantDesc>()
    );
};

/// Restore the producer's plane offset onto a tensor rebuilt from `desc`.
///
/// Applies only to a handle-based import. Under a host kind (`kind::MEM`,
/// `kind::SHM`) the descriptor's `ptr` is the pinned address of the view
/// itself and the offset is not applied.
fn apply_plane_offset(tensor: &mut TensorDyn, desc: &TensorDesc, plane_offset: u64) {
    if plane_offset == 0 || desc.is_host() {
        return;
    }
    tensor.set_plane_offset(plane_offset as usize);
}

/// Payload of the `edgefirst_tensor_v2` capsule: the wire format of the
/// cross-package tensor protocol.
///
/// A consumer in a different extension module reads only the `#[repr(C)]`
/// fields `desc`, `quant` and `plane_offset`. The trailing keepalive fields
/// are dropped by the producer's capsule destructor.
#[repr(C)]
pub struct TensorCapsulePayload {
    pub desc: TensorDesc,
    /// Quantization metadata, borrowing [`Self::quant_keepalive`]'s arrays.
    /// [`QuantDesc::absent`] when the tensor carries none.
    pub quant: QuantDesc,
    /// The producer's [`TensorDyn::plane_offset`]: the byte offset within
    /// the underlying buffer where this tensor's data starts. `0` for a
    /// whole tensor.
    pub plane_offset: u64,
    pub pin: Option<HostPin<'static>>,
    /// Owns the arrays [`Self::quant`] points at, for the capsule's life.
    pub quant_keepalive: Option<Arc<Quantization>>,
    /// Keepalive for a PBO-backed `desc`'s `ptr` (a `PboOpsVtable` address
    /// under `kind::PBO`). `None` for every kind but `PBO`.
    pub pbo_keepalive: Option<Arc<dyn Send + Sync>>,
}

/// A tensor argument resolved from an arbitrary Python object, valid for the
/// GIL lifetime `'py` of the `Bound` it was extracted from.
///
/// Obtained via [`TensorArg::extract`] (read-only, e.g. a `convert()`
/// source) or [`TensorArg::extract_mut`] (exclusive, e.g. a `convert()`
/// destination); read the access through the `AsRef`/`AsMut` impls below.
pub enum TensorArg<'py> {
    /// Same extension module: a runtime-borrow-checked reference into the
    /// caller's own `PyTensor`.
    NativeRef(PyRef<'py, crate::tensor::PyTensor>),
    /// Same extension module, exclusive access.
    NativeMut(PyRefMut<'py, crate::tensor::PyTensor>),
    /// A different extension module: imported via the capsule protocol.
    /// `_keepalive` is the producer's capsule, which owns the pin or vtable
    /// keepalive backing `tensor`'s `ptr` and must outlive every use of
    /// `tensor`.
    Foreign {
        tensor: Box<TensorDyn>,
        _keepalive: Py<PyAny>,
    },
}

impl<'py> TensorArg<'py> {
    /// Extract a read-only tensor argument, e.g. a `convert()` source.
    ///
    /// `access` is forwarded to `__edgefirst_tensor__` when the foreign-module
    /// path is taken: `None` requests the native handle only, `Some("read")`
    /// requests the host address.
    ///
    /// A `None` request that comes back host-kind with no address is retried
    /// once with `access="read"`.
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
    /// A `None` request that comes back host-kind with no address is retried
    /// once with `access="readwrite"`. An explicit `access` request is not
    /// retried.
    pub fn extract_mut(obj: &Bound<'py, PyAny>, access: Option<&str>) -> PyResult<Self> {
        #[cfg(feature = "codec")]
        if obj.cast::<crate::codec::PyFrame>().is_ok() {
            return Err(crate::codec::read_only("writing into it"));
        }
        if let Ok(native) = obj.cast::<crate::tensor::PyTensor>() {
            return Ok(Self::NativeMut(native.try_borrow_mut()?));
        }
        let retry_access = access.is_none().then_some("readwrite");
        Self::from_protocol(obj, access, retry_access)
    }

    /// Whether [`Self::into_raw_access`] can succeed for `self`.
    ///
    /// `Foreign` is always detachable. `NativeRef`/`NativeMut` are detachable
    /// when `reconstructible` holds for their backing memory, which excludes
    /// `Cuda`. A caller whose argument fails this check keeps the GIL held for
    /// the whole operation.
    pub fn can_detach(&self) -> bool {
        match self {
            Self::Foreign { .. } => true,
            Self::NativeRef(guard) => reconstructible(&guard.0),
            Self::NativeMut(guard) => reconstructible(&guard.0),
        }
    }

    /// The `__edgefirst_tensor__` capsule-protocol path shared by
    /// [`Self::extract`] and [`Self::extract_mut`].
    ///
    /// `retry_access`, when `Some`, is the access string to retry with if the
    /// first call comes back host-kind with a null `ptr`: `"read"` for
    /// [`Self::extract`], `"readwrite"` for [`Self::extract_mut`].
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

        let (mut capsule_obj, mut parts) = Self::call_protocol(&method, access)?;
        if let Some(retry_access) = retry_access {
            if parts.desc.is_host() && parts.desc.ptr.is_null() {
                // A producer that declared no reads is pinned write-only.
                let declared = edgefirst_tensor::protocol::flags::cpu_access(parts.desc.flags);
                let retry_access = match declared {
                    Some(a) if retry_access == "readwrite" => {
                        a.union(edgefirst_tensor::CpuAccess::Write).as_str()
                    }
                    _ => retry_access,
                };
                (capsule_obj, parts) = Self::call_protocol(&method, Some(retry_access))?;
            }
        }

        let mut tensor = TensorDyn::import_descriptor(&parts.desc).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "failed to import tensor via the __edgefirst_tensor__ protocol: {e}"
            ))
        })?;
        // Restore the quantization and the sub-region offset, which the
        // descriptor does not carry.
        apply_quantization(&mut tensor, parts.quant)?;
        apply_plane_offset(&mut tensor, &parts.desc, parts.plane_offset);

        Ok(Self::Foreign {
            tensor: Box::new(tensor),
            _keepalive: capsule_obj.unbind(),
        })
    }

    /// Call `__edgefirst_tensor__(access=...)`, validate the returned
    /// capsule, and copy out its descriptor.
    fn call_protocol(
        method: &Bound<'py, PyAny>,
        access: Option<&str>,
    ) -> PyResult<(Bound<'py, PyAny>, CapsuleParts)> {
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
                "__edgefirst_tensor__() must return a PyCapsule named \"edgefirst_tensor_v2\", \
                 per the cross-package tensor protocol",
            )
        })?;

        // `pointer_checked` validates the capsule name and returns the payload
        // pointer.
        let ptr = capsule
            .pointer_checked(Some(c"edgefirst_tensor_v2"))
            .map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "__edgefirst_tensor__() returned a capsule not named \
                     \"edgefirst_tensor_v2\". Either the object does not follow the \
                     cross-package tensor protocol, or it comes from an edgefirst.* \
                     package built against another capsule version \
                     -- install \
                     matching edgefirst.* versions",
                )
            })?;
        // SAFETY: a capsule named "edgefirst_tensor_v2" wraps a
        // `TensorCapsulePayload`, and `capsule_obj` is alive for the whole call.
        let parts = unsafe { read_capsule_parts(ptr.as_ptr() as *const TensorCapsulePayload) }?;

        Ok((capsule_obj, parts))
    }
}

/// A resolved tensor argument with every Python guard released, which the
/// holder may carry into [`Python::detach`].
///
/// [`TensorArg::into_raw_access`] reconstructs an independent `TensorDyn` for
/// the native variants through [`TensorDesc`] and
/// [`TensorDyn::import_descriptor`]. The reconstructed value refers to the
/// same underlying buffer as the original.
///
/// State a backend records on the reconstructed tensor, such as the fence
/// value of a D3D11 texture's last GPU write, is copied back onto the caller's
/// object by [`publish_gpu_write`].
pub struct RawTensorAccess {
    repr: RawTensorRepr,
}

enum RawTensorRepr {
    /// Read-only origin: `TensorArg::NativeRef`. `AsMut::as_mut` panics on it.
    Ref {
        tensor: TensorDyn,
        _pin: Option<HostPin<'static>>,
        _keepalive: Option<Py<PyAny>>,
    },
    /// Exclusive origin: `TensorArg::NativeMut` or `TensorArg::Foreign`.
    Mut {
        tensor: TensorDyn,
        _pin: Option<HostPin<'static>>,
        _keepalive: Option<Py<PyAny>>,
    },
}

// SAFETY: `TensorDyn`, `HostPin<'static>` and `Py<PyAny>` are each `Send`.
unsafe impl Send for RawTensorAccess {}

impl<'py> TensorArg<'py> {
    /// Resolve `self` into a [`RawTensorAccess`], dropping every Python guard.
    ///
    /// The native path reconstructs an independent `TensorDyn` and returns an
    /// error for an unrecognised descriptor kind or a pin the backend refuses.
    /// The foreign path does not fail.
    ///
    /// A caller that renders into the reconstructed tensor passes it to
    /// [`publish_gpu_write`] afterwards.
    pub fn into_raw_access(self) -> PyResult<RawTensorAccess> {
        use edgefirst_tensor::CpuAccess;

        Ok(match self {
            Self::NativeRef(guard) => {
                let (tensor, pin) = reconstruct(&guard.0, CpuAccess::Read)?;
                let py = guard.py();
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
                let (tensor, pin) = reconstruct(&guard.0, write_pin_access(&guard.0))?;
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

/// The pin direction for a tensor about to be written: its declaration plus
/// `Write`.
pub(crate) fn write_pin_access(tensor: &TensorDyn) -> edgefirst_tensor::CpuAccess {
    tensor
        .cpu_access()
        .union(edgefirst_tensor::CpuAccess::Write)
}

/// Build an independent `TensorDyn` aliasing the same backing memory as
/// `tensor`. Requests no pin unless the descriptor comes back host-kind with a
/// null `ptr`, in which case it pins with `access` and returns the pin.
pub(crate) fn reconstruct(
    tensor: &TensorDyn,
    access: edgefirst_tensor::CpuAccess,
) -> PyResult<(TensorDyn, Option<HostPin<'static>>)> {
    let desc = tensor.descriptor_pinned(None);
    let (desc, pin) = if desc.is_host() && desc.ptr.is_null() {
        let pin = tensor.pin_host(access).map_err(|e| {
            crate::tensor::runtime_or_permission_error(
                e.cpu_access_denied().is_some(),
                format!("failed to resolve a tensor for a detached region: {e}"),
            )
        })?;
        let desc = tensor.descriptor_pinned(Some(&pin));
        (desc, Some(pin))
    } else {
        (desc, None)
    };
    let mut imported = TensorDyn::import_descriptor(&desc).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "failed to resolve a tensor for a detached region: {e}"
        ))
    })?;
    // Copy the quantization and the sub-region offset from the source tensor.
    apply_quantization(&mut imported, tensor.quantization().cloned())?;
    apply_plane_offset(
        &mut imported,
        &desc,
        tensor.plane_offset().unwrap_or(0) as u64,
    );
    Ok((imported, pin))
}

/// The fence value a render recorded on the tensor it wrote.
///
/// `None` for a backing that is not a D3D11 texture, or a texture no render
/// has recorded a write on.
#[cfg(feature = "image")]
pub(crate) fn recorded_gpu_write(rendered: &TensorDyn) -> Option<u64> {
    #[cfg(target_os = "windows")]
    {
        if rendered.memory() != edgefirst_tensor::TensorMemory::DmaBuf {
            return None;
        }
        Some(rendered.gpu_write_value()).filter(|v| *v != 0)
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = rendered;
        None
    }
}

/// Record a render's completion on the object the caller passed as the
/// destination.
///
/// `value` comes from [`recorded_gpu_write`] on the tensor the render wrote.
/// It is passed to the object's `set_gpu_write` method, looked up by name.
/// Every borrow of the destination must be dropped before the call.
///
/// Does not fail: a `None` value, a destination without `set_gpu_write` and a
/// call that raises are debug-logged at most.
#[cfg(feature = "image")]
pub(crate) fn publish_gpu_write(dst: &Bound<'_, PyAny>, value: Option<u64>) {
    let Some(value) = value else {
        return;
    };
    let Ok(method) = dst.getattr("set_gpu_write") else {
        return;
    };
    if !method.is_callable() {
        return;
    }
    if let Err(e) = method.call1((value,)) {
        log::debug!("publishing a render completion onto the destination: {e}");
    }
}

/// Whether [`reconstruct`] can succeed for `tensor`'s backing memory. Only
/// `Cuda` is excluded.
pub(crate) fn reconstructible(tensor: &TensorDyn) -> bool {
    !matches!(tensor.memory(), edgefirst_tensor::TensorMemory::Cuda)
}

/// What keeps a tensor taken out of a [`RawTensorAccess`] valid: the host
/// pin and the Python object it was resolved from. Drop it after the tensor.
#[cfg(feature = "codec")]
pub struct RawKeepalive {
    _pin: Option<HostPin<'static>>,
    _keepalive: Option<Py<PyAny>>,
}

#[cfg(feature = "codec")]
impl RawTensorAccess {
    /// Takes the resolved tensor out, for a consumer that owns its tensors
    /// (a decoder's provided pool), with what must outlive it.
    pub fn into_owned(self) -> (TensorDyn, RawKeepalive) {
        match self.repr {
            RawTensorRepr::Ref {
                tensor,
                _pin,
                _keepalive,
            }
            | RawTensorRepr::Mut {
                tensor,
                _pin,
                _keepalive,
            } => (tensor, RawKeepalive { _pin, _keepalive }),
        }
    }
}

impl AsRef<TensorDyn> for RawTensorAccess {
    /// Borrow the resolved tensor.
    fn as_ref(&self) -> &TensorDyn {
        match &self.repr {
            RawTensorRepr::Ref { tensor, .. } => tensor,
            RawTensorRepr::Mut { tensor, .. } => tensor,
        }
    }
}

impl AsMut<TensorDyn> for RawTensorAccess {
    /// Mutably borrow the resolved tensor.
    ///
    /// # Panics
    ///
    /// Panics if `self` was resolved from `TensorArg::extract` (read-only)
    /// rather than `extract_mut`.
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
    /// than [`TensorArg::extract_mut`].
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

#[cfg(feature = "decoder")]
mod decoder_interop {
    use super::{Bound, Py, PyAny, PyAnyMethods, PyCapsule, PyRef, PyResult, PyTypeMethods};
    use pyo3::types::PyCapsuleMethods;

    /// Payload of the `edgefirst_decoder_v1` capsule.
    ///
    /// `decoder_size` and `decoder_align` are the size and alignment of the
    /// producer's `Decoder`, compared by [`DecoderArg::extract`].
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub(crate) struct DecoderCapsulePayload {
        pub(crate) ptr: usize,
        pub(crate) decoder_size: usize,
        pub(crate) decoder_align: usize,
    }

    /// A `Decoder` argument resolved from an arbitrary Python object, valid
    /// for the GIL lifetime `'py` of the `Bound` it was extracted from.
    pub enum DecoderArg<'py> {
        /// Same extension module: a live, runtime-borrow-checked reference
        /// into the caller's own `PyDecoder`.
        Native(PyRef<'py, crate::decoder::PyDecoder>),
        /// A different extension module: a borrowed pointer imported via
        /// the layout-guarded `__edgefirst_decoder__` capsule protocol.
        /// `_keepalive` is the producer object, which keeps the `Decoder` that
        /// `decoder` points into alive.
        ///
        /// The reference bypasses PyO3's runtime borrow check on the
        /// producer's `PyDecoder`. The GIL must stay held while it is in use;
        /// a caller detaches only for `DecoderArg::Native`.
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
            // SAFETY: a capsule named "edgefirst_decoder_v1" wraps a
            // `DecoderCapsulePayload`, which is `Copy`.
            let payload: DecoderCapsulePayload =
                unsafe { *(ptr.as_ptr() as *const DecoderCapsulePayload) };

            // Layout guard: the producer and consumer must agree on `Decoder`'s
            // size and alignment.
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

            // SAFETY: the producer and consumer agree on `Decoder`'s size and
            // alignment, and `_keepalive` retains the producer object `decoder`
            // points into.
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
        Bound, HostPin, Py, PyAny, PyAnyMethods, PyCapsule, PyResult, PyTypeMethods, TensorDyn,
    };
    #[cfg(feature = "decoder")]
    use super::{IntoPyObject, PyRef};
    use edgefirst_tensor::ProtoData;
    use pyo3::types::PyCapsuleMethods;

    /// A `ProtoData` argument resolved from an arbitrary Python object.
    ///
    /// The foreign path rebuilds the two tensors through the
    /// `__edgefirst_tensor__` protocol.
    pub enum ProtoDataArg<'py> {
        /// Same extension module.
        #[cfg(feature = "decoder")]
        Native(PyRef<'py, crate::decoder::PyProtoData>),
        /// A different extension module: an owned `ProtoData` reconstructed
        /// from two `__edgefirst_tensor__` capsules plus a layout code.
        ///
        /// `_keepalive` retains both producer capsules, which hold the pins
        /// backing `proto_data`'s tensors.
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

    /// A `ProtoData` resolved for `py.detach`: both tensors are reconstructed
    /// into independent values, as in [`super::TensorArg::into_raw_access`].
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

    // SAFETY: the two reconstructed `TensorDyn`s and the keepalive's
    // `Py<PyAny>`s are `Send`.
    unsafe impl Send for RawProtoDataAccess {}

    impl<'py> ProtoDataArg<'py> {
        /// Whether [`Self::into_raw_access`] can succeed for `self`: both
        /// `mask_coefficients` and `protos` must be detachable.
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
        /// Python guard. Callers check [`Self::can_detach`] first.
        pub fn into_raw_access(self) -> PyResult<RawProtoDataAccess> {
            #[cfg(feature = "decoder")]
            use edgefirst_tensor::CpuAccess;

            match self {
                #[cfg(feature = "decoder")]
                Self::Native(guard) => {
                    // Both tensors are reconstructed read-only.
                    let (mask_coefficients, mask_pin) =
                        super::reconstruct(&guard.0.mask_coefficients, CpuAccess::Read)?;
                    let (protos, protos_pin) =
                        super::reconstruct(&guard.0.protos, CpuAccess::Read)?;
                    let layout = guard.0.layout;
                    let py = guard.py();
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
        /// Borrow the resolved proto data.
        fn as_ref(&self) -> &ProtoData {
            &self.proto_data
        }
    }

    /// Validate and import a single `edgefirst_tensor_v2` capsule.
    fn import_tensor_capsule(capsule_obj: &Bound<'_, PyAny>) -> PyResult<TensorDyn> {
        let capsule = capsule_obj.cast::<PyCapsule>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "__edgefirst_protodata__() must return PyCapsule objects named \
                 \"edgefirst_tensor_v2\", per the cross-package tensor protocol",
            )
        })?;
        let ptr = capsule
            .pointer_checked(Some(c"edgefirst_tensor_v2"))
            .map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "__edgefirst_protodata__() returned a capsule not named \
                     \"edgefirst_tensor_v2\". Either the object does not follow the \
                     cross-package tensor protocol, or it comes from an edgefirst.* \
                     package built against another capsule version \
                     -- install \
                     matching edgefirst.* versions",
                )
            })?;
        // SAFETY: a capsule named "edgefirst_tensor_v2" wraps a
        // `TensorCapsulePayload`, and `capsule_obj` is alive for the call.
        let parts = unsafe {
            super::read_capsule_parts(ptr.as_ptr() as *const super::TensorCapsulePayload)
        }?;
        let mut tensor = TensorDyn::import_descriptor(&parts.desc).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "failed to import a ProtoData tensor via the __edgefirst_tensor__ protocol: {e}"
            ))
        })?;
        super::apply_quantization(&mut tensor, parts.quant)?;
        super::apply_plane_offset(&mut tensor, &parts.desc, parts.plane_offset);
        Ok(tensor)
    }
}
#[cfg(feature = "decoder")]
pub use decoder_interop::DecoderArg;
#[cfg(feature = "decoder")]
pub(crate) use decoder_interop::DecoderCapsulePayload;
#[cfg(any(feature = "image", feature = "decoder"))]
pub use proto_interop::{ProtoDataArg, RawProtoDataAccess};
