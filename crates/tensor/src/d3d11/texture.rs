// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//! `ID3D11Texture2D`-backed storage. GPU consumers read the texture directly;
//! the CPU reads and writes through a staging copy, or, for write-only
//! tensors, through a host buffer uploaded with `UpdateSubresource`.
//!
//! The host buffer is also what [`D3d11TextureTensor::host_pin`] hands out an
//! address into, for every access kind: a pin outlives any map, so it cannot
//! be the staging mapping itself (that would hold the staging texture mapped
//! forever and freeze the data). `sync_for_cpu` and `sync_for_device` move
//! bytes between that buffer and the texture.

use super::com::hr;
use super::device::{
    device, duplicate_handle, duplicate_raw_handle, wait_cpu_for, D3d11Device, GpuCompletion,
};
use crate::d3d11_layout::{image_d3d11_layout, D3d11ImageLayout};
use crate::{
    BufferIdentity, CpuAccess, DType, Error, IdentityKind, PixelFormat, PixelLayout, Result,
    TensorMemory, TensorTrait,
};
use num_traits::Num;
use std::cell::UnsafeCell;
use std::ffi::c_void;
use std::fmt;
use std::marker::PhantomData;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle, RawHandle};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use windows::core::{IUnknown, Interface, PCWSTR};
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Fence, ID3D11Texture2D, D3D11_BIND_RENDER_TARGET, D3D11_BIND_SHADER_RESOURCE,
    D3D11_CPU_ACCESS_READ, D3D11_CPU_ACCESS_WRITE, D3D11_MAPPED_SUBRESOURCE,
    D3D11_MAP_FLAG_DO_NOT_WAIT, D3D11_MAP_READ, D3D11_MAP_READ_WRITE, D3D11_RESOURCE_MISC_SHARED,
    D3D11_RESOURCE_MISC_SHARED_NTHANDLE, D3D11_TEXTURE2D_DESC, D3D11_USAGE, D3D11_USAGE_DEFAULT,
    D3D11_USAGE_STAGING,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::Dxgi::{
    IDXGIResource1, DXGI_ERROR_WAS_STILL_DRAWING, DXGI_SHARED_RESOURCE_READ,
    DXGI_SHARED_RESOURCE_WRITE,
};

/// Bind flags every allocated texture carries: a render target so the image
/// crate can draw into it, a shader resource so it can be sampled from.
const BIND: u32 = (D3D11_BIND_RENDER_TARGET.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32;
/// Shared with an NT handle, which is the form every cross-API consumer
/// (D3D12, CUDA, ANGLE) opens.
const MISC: u32 = (D3D11_RESOURCE_MISC_SHARED.0 | D3D11_RESOURCE_MISC_SHARED_NTHANDLE.0) as u32;

/// Who holds a CPU window onto a backing right now.
///
/// `map_with` and `scoped_pin` take `&self`, and views share their parent's
/// backing, so two maps of one staging texture or one host buffer are
/// reachable from safe code. Without this they would produce a second `Map`
/// on a mapped subresource, a `CopyResource` into a mapped staging texture,
/// or two `&mut` aliases into one buffer. Same shape as `PboTensor`'s
/// `map_state`: readers refcount behind one mapping, a writer is exclusive.
#[derive(Default)]
struct MapState {
    /// Live read maps sharing one mapping.
    readers: usize,
    /// A writable map is live. Never true while `readers > 0`.
    writer: bool,
    /// Base address of the live mapping, meaningful only while `readers > 0`
    /// or `writer`. Stored as an integer so `MapState` stays `Send`.
    base: usize,
    /// A `CopyResource` into the staging texture is queued and no `Map` has
    /// consumed it yet, so a retry must not queue another one.
    refresh_pending: bool,
}

impl MapState {
    /// Claims a window. `Ok(Some(base))` means this caller joined a live read
    /// mapping; `Ok(None)` means it is the one that must establish the
    /// mapping and record `base`.
    fn claim(&mut self, writes: bool) -> Result<Option<usize>> {
        if self.writer || (writes && self.readers > 0) {
            let held = if self.writer { "write" } else { "read" };
            let wanted = if writes { "write" } else { "read" };
            return Err(Error::InvalidOperation(format!(
                "D3D11 texture tensor is already mapped for {held}; drop that map before taking a {wanted} map"
            )));
        }
        if writes {
            self.writer = true;
            return Ok(None);
        }
        if self.readers > 0 {
            self.readers += 1;
            return Ok(Some(self.base));
        }
        self.readers = 1;
        Ok(None)
    }

    /// Gives a claim back. Returns whether that was the last holder, so the
    /// caller knows to tear the mapping down.
    fn release(&mut self, writes: bool) -> bool {
        if writes {
            self.writer = false;
        } else {
            self.readers -= 1;
        }
        let last = !self.writer && self.readers == 0;
        if last {
            self.base = 0;
        }
        last
    }
}

/// The one staging texture behind every readable map of a texture (views
/// share it). `row_pitch` is the driver's, queried once at creation.
struct Staging {
    tex: ID3D11Texture2D,
    row_pitch: usize,
    rows: usize,
    state: Mutex<MapState>,
}

impl Staging {
    fn bytes(&self) -> usize {
        self.row_pitch * self.rows
    }

    fn state(&self) -> std::sync::MutexGuard<'_, MapState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Host rows the storage owns for as long as the tensor lives: the address
/// [`D3d11TextureTensor::host_pin`] hands out, and the bytes a write-only
/// tensor's maps write through. `row_pitch` is the backing pitch it was
/// allocated at, which is also the source pitch of its upload.
struct HostBuffer {
    /// The allocation, over-sized by `HOST_PIN_ALIGN - 1` so the rows can
    /// start at an aligned address inside it. Never borrowed again after
    /// construction; it is here to own the bytes [`Self::base`] points at.
    _bytes: UnsafeCell<Box<[u8]>>,
    /// Aligned address of the first row, taken once in [`Self::new`]. Taking
    /// it per call would mint a fresh `&mut` over the allocation while pins
    /// and maps still hold pointers derived from an earlier one, which the
    /// aliasing model does not allow.
    base: *mut u8,
    /// Bytes of rows, excluding the alignment slack.
    len: usize,
    row_pitch: usize,
    state: Mutex<MapState>,
}

/// Alignment of the address [`D3d11TextureTensor::host_pin`] hands out.
///
/// `crate::pin::HostPin::alignment` documents 64 bytes as what TFLite's
/// `SetCustomAllocationForTensor` requires, and every other backing satisfies
/// it by being page-backed. Rust's allocator promises only `align_of::<u8>()`,
/// so this backing has to ask.
const HOST_PIN_ALIGN: usize = 64;

// SAFETY: the buffer is written by one mapper at a time -- `MapState` makes a
// writer exclusive -- and read by an upload only under the same claim, which
// `HostBufferGuard::drop` holds across it. `host_pin` is the one path that
// takes no claim at all: it hands out the address for the tensor's lifetime,
// so a pin holder carries the coherency obligation `crate::pin` documents for
// every backend, and this type does not arbitrate it.
unsafe impl Sync for HostBuffer {}
// SAFETY: a `Box<[u8]>` has no thread affinity; see the `Sync` note above for
// the access discipline.
unsafe impl Send for HostBuffer {}

impl HostBuffer {
    fn new(row_pitch: usize, rows: usize) -> Self {
        let len = row_pitch * rows;
        let mut bytes = vec![0u8; len + HOST_PIN_ALIGN - 1].into_boxed_slice();
        let start = bytes.as_mut_ptr();
        // At most `HOST_PIN_ALIGN - 1`, which is exactly the slack allocated.
        // SAFETY: the offset is inside the allocation by construction.
        let base = unsafe { start.add(start.align_offset(HOST_PIN_ALIGN)) };
        HostBuffer {
            _bytes: UnsafeCell::new(bytes),
            base,
            len,
            row_pitch,
            state: Mutex::new(MapState::default()),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, MapState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Aligned base address and length of the rows, excluding the slack.
    fn span(&self) -> (*mut u8, usize) {
        (self.base, self.len)
    }
}

pub struct D3d11TextureTensor<T> {
    device: &'static D3d11Device,
    texture: ID3D11Texture2D,
    layout: D3d11ImageLayout,
    desc_bind_flags: u32,
    width: usize,
    height: usize,
    format: PixelFormat,
    dtype: DType,
    shape: Vec<usize>,
    name: String,
    access: CpuAccess,
    staging: Option<Arc<Staging>>,
    /// Created on the first pin or the first map of a write-only tensor, and
    /// shared with every view: the `Arc` is the cell, not the buffer, so a
    /// view made before the first pin still lands on the same buffer.
    host: Arc<OnceLock<HostBuffer>>,
    nt_handle: Arc<OwnedHandle>,
    identity: BufferIdentity,
    last_gpu_write: Arc<AtomicU64>,
    /// The producer's fence, opened at import and kept for the tensor's
    /// lifetime because the immediate context holds a queued `Wait` on it.
    /// `None` for an allocated or wrapped texture, and for an import that
    /// carried no completion.
    imported_fence: Option<ID3D11Fence>,
    /// Set when a caller took a writable window on the host buffer, so
    /// `sync_for_device` knows which backing carries its writes.
    wrote_host: Arc<AtomicBool>,
    view_offset: usize,
    _marker: PhantomData<T>,
}

/// How long an import blocks on a producer's fence when this device cannot
/// queue a GPU-side wait. Bounded rather than infinite so a value the producer
/// never signals costs a warning instead of a hung import.
const IMPORT_WAIT_MS: u32 = 5_000;

fn texture_desc(
    layout: &D3d11ImageLayout,
    usage: D3D11_USAGE,
    bind: u32,
    cpu: u32,
    misc: u32,
) -> D3D11_TEXTURE2D_DESC {
    D3D11_TEXTURE2D_DESC {
        Width: layout.texture_width as u32,
        Height: layout.texture_height as u32,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT(layout.dxgi_format as i32),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: usage,
        BindFlags: bind,
        CPUAccessFlags: cpu,
        MiscFlags: misc,
    }
}

fn create_texture(d: &D3d11Device, desc: &D3D11_TEXTURE2D_DESC) -> Result<ID3D11Texture2D> {
    let mut out: Option<ID3D11Texture2D> = None;
    // SAFETY: `desc` is fully initialised and `out` is a valid out-parameter.
    hr("ID3D11Device::CreateTexture2D", unsafe {
        d.dev().CreateTexture2D(desc, None, Some(&mut out))
    })?;
    out.ok_or_else(|| {
        Error::IoError(std::io::Error::other(
            "ID3D11Device::CreateTexture2D returned no texture",
        ))
    })
}

/// The texture's shared NT handle, owned by this process. Fails on a texture
/// created without `D3D11_RESOURCE_MISC_SHARED_NTHANDLE`.
fn create_nt_handle(tex: &ID3D11Texture2D) -> Result<OwnedHandle> {
    let res = hr(
        "ID3D11Texture2D::QueryInterface(IDXGIResource1)",
        tex.cast::<IDXGIResource1>(),
    )?;
    // SAFETY: `res` is live; an unnamed handle with read and write access.
    let handle = hr("IDXGIResource1::CreateSharedHandle", unsafe {
        res.CreateSharedHandle(
            None,
            (DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE).0,
            PCWSTR::null(),
        )
    })?;
    // SAFETY: `CreateSharedHandle` transferred ownership of `handle` to us.
    Ok(unsafe { OwnedHandle::from_raw_handle(handle.0) })
}

/// A staging texture of this layout, readable and writable by the CPU.
fn create_staging_texture(d: &D3d11Device, layout: &D3d11ImageLayout) -> Result<ID3D11Texture2D> {
    let cpu = (D3D11_CPU_ACCESS_READ.0 | D3D11_CPU_ACCESS_WRITE.0) as u32;
    create_texture(d, &texture_desc(layout, D3D11_USAGE_STAGING, 0, cpu, 0))
}

/// The driver's row pitch for a staging texture, learned from one throwaway
/// map. The pitch is chosen when the texture is created, so reading it takes a
/// texture; no query answers from a description alone.
fn staging_row_pitch(d: &D3d11Device, tex: &ID3D11Texture2D) -> Result<usize> {
    let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
    // SAFETY: `tex` is a live staging texture and `mapped` is a valid
    // out-parameter; a read map purely to learn the pitch.
    hr("ID3D11DeviceContext::Map (pitch query)", unsafe {
        d.ctx().Map(tex, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
    })?;
    let row_pitch = mapped.RowPitch as usize;
    // SAFETY: `tex` is the texture mapped on the line above.
    unsafe { d.ctx().Unmap(tex, 0) };
    Ok(row_pitch)
}

/// The staging texture readable maps go through, plus the driver's row pitch
/// for it.
fn create_staging(d: &D3d11Device, layout: &D3d11ImageLayout) -> Result<Staging> {
    let tex = create_staging_texture(d, layout)?;
    let row_pitch = staging_row_pitch(d, &tex)?;
    Ok(Staging {
        tex,
        row_pitch,
        rows: layout.texture_height,
        state: Mutex::new(MapState::default()),
    })
}

/// A semi-planar layout widened to the row pitch the driver gives its staging
/// copy, so the texture's texel grid and the combined plane's row stride are
/// one number.
///
/// The HAL's semi-planar model is a linear combined plane whose row pitch is
/// the sampled texture's texel width: CPU producers write luma and chroma
/// lines at the row stride, and the Path B shader wraps chroma addressing at
/// that same width. A texture only `even(width)` texels wide breaks that
/// wherever the driver pads the pitch -- 128 bytes on this NVIDIA adapter --
/// because one producer row then spans two texture rows and the shader samples
/// past the row edge.
///
/// The pitch is fixed when a texture is created, so learning it costs an
/// allocation: one staging texture at the floor width, and, when that came
/// back padded, a second at the pitch to confirm the wider texture is not
/// padded again. Every `CpuAccess` pays for the probe, the ones with no
/// staging of their own included, so a GPU-only tensor's texture has the same
/// geometry as a readable one's.
fn widen_semi_planar_to_pitch(
    d: &D3d11Device,
    layout: D3d11ImageLayout,
) -> Result<D3d11ImageLayout> {
    // R8, so a pitch in bytes is a pitch in texels.
    let pitch = staging_row_pitch(d, &create_staging_texture(d, &layout)?)?;
    if pitch <= layout.tight_row_bytes() {
        return Ok(layout);
    }
    // Rounded up because a chroma pair cannot straddle a row edge. Every pitch
    // either adapter has produced here is a multiple of four, so the rounding
    // has never moved the number; it holds the invariant, it does not measure.
    let widened = layout.widened_to(pitch.next_multiple_of(2));
    let second = staging_row_pitch(d, &create_staging_texture(d, &widened)?)?;
    if second != widened.tight_row_bytes() {
        return Err(Error::InvalidOperation(format!(
            "D3D11 semi-planar allocation: a {}-texel-wide R8 staging texture still has a row \
             pitch of {second}, so no texture width equals the pitch a CPU map sees",
            widened.texture_width
        )));
    }
    Ok(widened)
}

/// Process-local key for a texture the HAL allocated or wrapped.
fn tex_key(tex: &ID3D11Texture2D) -> u64 {
    tex.as_raw() as usize as u64
}

/// Checks an externally created texture against the layout the HAL would have
/// chosen, and returns that layout with the texture's own bind flags.
/// `BIND_RENDER_TARGET` is not required here; the image crate checks it when
/// the tensor is a destination.
fn validate_external(
    d: &D3d11Device,
    tex: &ID3D11Texture2D,
    width: usize,
    height: usize,
    format: PixelFormat,
    dtype: DType,
) -> Result<(D3d11ImageLayout, u32)> {
    let layout = image_d3d11_layout(format, dtype, width, height).ok_or_else(|| {
        Error::InvalidArgument(format!(
            "external D3D11 texture: no zero-copy D3D11 texture layout exists for \
             {format:?}/{dtype:?} {width}x{height}"
        ))
    })?;
    let mut desc = D3D11_TEXTURE2D_DESC::default();
    // SAFETY: `tex` is live and `desc` is a valid out-parameter.
    unsafe { tex.GetDesc(&mut desc) };
    if desc.Usage != D3D11_USAGE_DEFAULT {
        return Err(Error::InvalidArgument(format!(
            "external D3D11 texture: Usage is {} but D3D11_USAGE_DEFAULT is required",
            desc.Usage.0
        )));
    }
    if desc.BindFlags & D3D11_BIND_SHADER_RESOURCE.0 as u32 == 0 {
        return Err(Error::InvalidArgument(format!(
            "external D3D11 texture: BindFlags {:#x} lack D3D11_BIND_SHADER_RESOURCE",
            desc.BindFlags
        )));
    }
    if desc.MipLevels != 1 {
        return Err(Error::InvalidArgument(format!(
            "external D3D11 texture: MipLevels is {} but 1 is required",
            desc.MipLevels
        )));
    }
    if desc.ArraySize != 1 {
        return Err(Error::InvalidArgument(format!(
            "external D3D11 texture: ArraySize is {} but 1 is required",
            desc.ArraySize
        )));
    }
    if desc.SampleDesc.Count != 1 {
        return Err(Error::InvalidArgument(format!(
            "external D3D11 texture: SampleDesc.Count is {} but 1 is required (no multisampling)",
            desc.SampleDesc.Count
        )));
    }
    if desc.Format.0 as u32 != layout.dxgi_format {
        return Err(Error::InvalidArgument(format!(
            "external D3D11 texture: Format is {} but {format:?}/{dtype:?} needs DXGI format {}",
            desc.Format.0, layout.dxgi_format
        )));
    }
    if desc.Height as usize != layout.texture_height {
        return Err(Error::InvalidArgument(format!(
            "external D3D11 texture: texture is {} rows tall but {format:?}/{dtype:?} \
             {width}x{height} needs {}",
            desc.Height, layout.texture_height
        )));
    }
    // A semi-planar texture carries the image's row stride as its width, so a
    // width above the image's even row is padding the host allocated, and it
    // becomes the tensor's backing pitch. Every other format's texture row is
    // its image row exactly, so a different width is a different image.
    let layout = if format.layout() == PixelLayout::SemiPlanar {
        let texture_width = desc.Width as usize;
        if texture_width < layout.texture_width {
            return Err(Error::InvalidArgument(format!(
                "external D3D11 texture: texture is {texture_width} texels wide but \
                 {format:?}/{dtype:?} {width}x{height} needs at least {} (the image's even row)",
                layout.texture_width
            )));
        }
        // The wrapped texture has to satisfy the same rule `new_image` enforces
        // on its own allocations, and for the same reason: maps and pins of
        // this tensor are laid out at the driver's staging pitch, so a texture
        // narrower than that pitch would be reported at one width and
        // addressed at another. A third-party 64-wide NV12 texture on this
        // NVIDIA adapter is exactly that case -- its staging pitch is 128.
        // The pitch is a property of a created texture, so it takes a probe;
        // the probe is dropped here and `finish` creates the real staging.
        let wrapped = layout.widened_to(texture_width);
        // Two rules, two messages: a chroma pair cannot straddle a row edge,
        // and the texel grid has to be the row stride a CPU map is laid out
        // at. An odd width fails the first whatever its pitch turns out to be.
        if !texture_width.is_multiple_of(2) {
            return Err(Error::InvalidArgument(format!(
                "external D3D11 texture: a {format:?} texture is {texture_width} texels wide, \
                 which is odd; a semi-planar row holds whole chroma pairs, so the width must \
                 be even"
            )));
        }
        let pitch = staging_row_pitch(d, &create_staging_texture(d, &wrapped)?)?;
        if pitch != texture_width {
            return Err(Error::InvalidArgument(format!(
                "external D3D11 texture: a {format:?} texture is {texture_width} texels wide with \
                 a staging row pitch of {pitch}; semi-planar textures must be as wide as the \
                 driver row pitch; allocate through the HAL or match the pitch"
            )));
        }
        wrapped
    } else {
        if desc.Width as usize != layout.texture_width {
            return Err(Error::InvalidArgument(format!(
                "external D3D11 texture: texture is {} texels wide but {format:?}/{dtype:?} \
                 {width}x{height} needs {}",
                desc.Width, layout.texture_width
            )));
        }
        layout
    };
    // SAFETY: `tex` is live.
    let owner = hr("ID3D11Texture2D::GetDevice", unsafe { tex.GetDevice() })?;
    let owner = hr(
        "ID3D11Device::QueryInterface(IUnknown)",
        owner.cast::<IUnknown>(),
    )?;
    let ours = hr(
        "ID3D11Device::QueryInterface(IUnknown)",
        d.dev().cast::<IUnknown>(),
    )?;
    if owner != ours {
        return Err(Error::InvalidArgument(
            "external D3D11 texture: GetDevice is not the HAL's process device".into(),
        ));
    }
    Ok((layout, desc.BindFlags))
}

impl<T> D3d11TextureTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    /// Allocates a texture on the process device for `format`/`dtype` at
    /// `width` x `height`, with the CPU staging `access` implies.
    pub(crate) fn new_image(
        width: usize,
        height: usize,
        format: PixelFormat,
        dtype: DType,
        shape: &[usize],
        name: Option<&str>,
        access: CpuAccess,
    ) -> Result<Self> {
        let layout = image_d3d11_layout(format, dtype, width, height).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "Tensor::image: no zero-copy D3D11 texture layout exists for {format:?}/{dtype:?} \
                 {width}x{height} (see edgefirst_tensor::d3d11_layout); pass memory=None to \
                 auto-select or Some(TensorMemory::Mem) for a CPU tensor"
            ))
        })?;
        let d = device()?;
        // A semi-planar texture has to be as wide as the row pitch a CPU map
        // sees; every other format's texture row already is its image row.
        let layout = if format.layout() == PixelLayout::SemiPlanar {
            widen_semi_planar_to_pitch(d, layout)?
        } else {
            layout
        };
        let tex = create_texture(
            d,
            &texture_desc(&layout, D3D11_USAGE_DEFAULT, BIND, 0, MISC),
        )?;
        let handle = create_nt_handle(&tex)?;
        let key = tex_key(&tex);
        Self::finish(
            d, tex, BIND, layout, width, height, format, dtype, shape, name, access, handle, key,
        )
    }

    /// Wraps a texture created on the HAL device, after checking its
    /// description against the layout the HAL would have chosen.
    ///
    /// # Safety
    ///
    /// `texture` must be null or a live `ID3D11Texture2D*`. Ownership stays
    /// with the caller: this takes its own reference.
    #[allow(clippy::too_many_arguments)] // one image description, spelled out
    pub(crate) unsafe fn from_d3d11_texture(
        texture: *mut c_void,
        width: usize,
        height: usize,
        format: PixelFormat,
        dtype: DType,
        shape: &[usize],
        access: CpuAccess,
        name: Option<&str>,
    ) -> Result<Self> {
        let d = device()?;
        // SAFETY: the caller guarantees `texture` is null or a live
        // `ID3D11Texture2D`; borrowing takes no reference and `cloned` AddRefs
        // the one this tensor keeps.
        let tex = unsafe { ID3D11Texture2D::from_raw_borrowed(&texture) }
            .cloned()
            .ok_or_else(|| Error::InvalidArgument("from_d3d11_texture: null texture".into()))?;
        let (layout, bind) = validate_external(d, &tex, width, height, format, dtype)?;
        let handle = create_nt_handle(&tex).map_err(|e| {
            Error::InvalidArgument(format!(
                "from_d3d11_texture: the texture was not created with \
                 D3D11_RESOURCE_MISC_SHARED_NTHANDLE: {e}"
            ))
        })?;
        let key = tex_key(&tex);
        Self::finish(
            d, tex, bind, layout, width, height, format, dtype, shape, name, access, handle, key,
        )
    }

    /// Opens a shared texture on the HAL device and, when a completion is
    /// given, waits for it on the immediate context so same-device readers
    /// need no further ordering.
    ///
    /// # Safety
    ///
    /// `handle` must be a shared NT handle valid in this process, and the
    /// completion's handle, when given, a shared fence handle. Both stay owned
    /// by the caller: this duplicates what it keeps.
    #[allow(clippy::too_many_arguments)] // one image description, spelled out
    pub(crate) unsafe fn from_d3d11_shared_handle(
        handle: RawHandle,
        width: usize,
        height: usize,
        format: PixelFormat,
        dtype: DType,
        shape: &[usize],
        access: CpuAccess,
        completion: Option<(RawHandle, u64)>,
        name: Option<&str>,
    ) -> Result<Self> {
        let d = device()?;
        let dev1 = d.dev1().ok_or_else(|| {
            Error::NotImplemented("ID3D11Device1 required to open shared handles".into())
        })?;
        // SAFETY: the caller guarantees an NT handle valid in this process.
        let tex: ID3D11Texture2D = hr("ID3D11Device1::OpenSharedResource1", unsafe {
            dev1.OpenSharedResource1(HANDLE(handle))
        })?;
        let (layout, bind) = validate_external(d, &tex, width, height, format, dtype)?;
        let own = duplicate_raw_handle(handle).map_err(Error::IoError)?;
        // The texture this open produced, not the handle value the caller
        // passed in. The caller may close its handle the moment this returns
        // -- the blob importer does exactly that -- and Windows hands the
        // freed value straight to the next `DuplicateHandle` in the process,
        // so an identity keyed on any handle value can be reused by an
        // unrelated later import and the image crate's EGLImage cache would
        // then serve the wrong texture. A COM object's address cannot be
        // reused that way: the cache's EGLImage holds its own reference to
        // the texture, so the address stays taken for as long as any entry
        // keyed on it can be found.
        let key = tex_key(&tex);
        let mut t = Self::finish(
            d, tex, bind, layout, width, height, format, dtype, shape, name, access, own, key,
        )?;
        if let Some((fence_handle, value)) = completion {
            // SAFETY: the caller guarantees a shared fence handle valid in
            // this process.
            unsafe { t.order_behind(d, fence_handle, value) }?;
        }
        Ok(t)
    }

    /// Orders this process's device behind the producer's recorded write and
    /// re-records the completion on the local timeline.
    ///
    /// `last_gpu_write` is a value on *this* device's fence everywhere else in
    /// this module: [`gpu_completion`](Self::gpu_completion) pairs it with
    /// `fence_shared_handle()`, and [`try_init_cuda`](Self::try_init_cuda)
    /// imports that same fence as the CUDA external semaphore. Storing the
    /// producer's value would break that pairing -- in-process it is harmless
    /// because the rendezvous makes both copies share one fence, but across
    /// processes the two timelines are unrelated, so a CUDA map would wait on
    /// the local fence for a value it may never reach and the importing thread
    /// would block indefinitely.
    ///
    /// So the producer's fence is opened and waited on, and a fresh local
    /// signal taken *after* that wait becomes the recorded completion. Every
    /// local consumer -- CPU maps through the immediate context, the CUDA
    /// semaphore wait, a re-export through a descriptor or a blob -- then
    /// works on one timeline. In-process the two fences are the same object,
    /// so the wait is one the driver satisfies immediately.
    ///
    /// A device that cannot signal has no local value to record, so it blocks
    /// on the producer's fence here instead: the same guarantee, paid for at
    /// import rather than at first use.
    ///
    /// # Safety
    ///
    /// `fence_handle` must be a shared fence handle valid in this process.
    /// Ownership stays with the caller.
    unsafe fn order_behind(
        &mut self,
        d: &'static D3d11Device,
        fence_handle: RawHandle,
        value: u64,
    ) -> Result<()> {
        let Some(dev5) = d.dev5() else {
            log::warn!(
                "D3D11 import: no ID3D11Device5 on the process device, so the producer's fence \
                 value {value} cannot be opened or waited on; the tensor records no completion"
            );
            return Ok(());
        };
        let mut fence: Option<ID3D11Fence> = None;
        // SAFETY: the caller guarantees a shared fence handle valid in this
        // process; `fence` is a valid out-parameter.
        hr("ID3D11Device5::OpenSharedFence", unsafe {
            dev5.OpenSharedFence(HANDLE(fence_handle), &mut fence)
        })?;
        let fence = fence.ok_or_else(|| {
            Error::IoError(std::io::Error::other(
                "ID3D11Device5::OpenSharedFence returned no fence",
            ))
        })?;
        let Some(ctx4) = d.ctx4().filter(|_| d.signal_supported()) else {
            if let Err(e) = wait_cpu_for(&fence, value, IMPORT_WAIT_MS) {
                log::warn!(
                    "D3D11 import: this device cannot queue a wait or signal a fence, and \
                     blocking on the producer's value {value} failed ({e}); the texture may \
                     still be being written"
                );
            }
            return Ok(());
        };
        // SAFETY: both interfaces are live; a GPU-side wait on the producer's
        // value, which every command queued after it is ordered behind.
        hr("ID3D11DeviceContext4::Wait", unsafe {
            ctx4.Wait(&fence, value)
        })?;
        // Kept for the tensor's lifetime: the queued wait names this fence
        // until it retires, and the tensor outlives that by construction.
        self.imported_fence = Some(fence);
        self.last_gpu_write
            .store(d.signal().unwrap_or(0), Ordering::Release);
        Ok(())
    }

    /// Shared tail of the three constructors: builds the CPU-side backing
    /// `access` calls for and assembles the tensor.
    #[allow(clippy::too_many_arguments)] // one image description, spelled out
    fn finish(
        d: &'static D3d11Device,
        texture: ID3D11Texture2D,
        bind_flags: u32,
        layout: D3d11ImageLayout,
        width: usize,
        height: usize,
        format: PixelFormat,
        dtype: DType,
        shape: &[usize],
        name: Option<&str>,
        access: CpuAccess,
        handle: OwnedHandle,
        key: u64,
    ) -> Result<Self> {
        // Only a readable tensor needs the staging texture. A write-only one
        // reaches the texture through the host buffer, and a hardware-only one
        // never reaches it from the CPU at all.
        let staging = match access {
            CpuAccess::Read | CpuAccess::ReadWrite => Some(Arc::new(create_staging(d, &layout)?)),
            CpuAccess::None | CpuAccess::Write => None,
        };
        Ok(Self {
            device: d,
            texture,
            layout,
            desc_bind_flags: bind_flags,
            width,
            height,
            format,
            dtype,
            shape: shape.to_vec(),
            name: name.unwrap_or("d3d11_tensor").to_owned(),
            access,
            staging,
            host: Arc::new(OnceLock::new()),
            nt_handle: Arc::new(handle),
            identity: BufferIdentity::derived(IdentityKind::D3d11Texture, key),
            last_gpu_write: Arc::new(AtomicU64::new(0)),
            imported_fence: None,
            wrote_host: Arc::new(AtomicBool::new(false)),
            view_offset: 0,
            _marker: PhantomData,
        })
    }

    fn require_access(&self, access: CpuAccess) -> Result<()> {
        if self.access == CpuAccess::None {
            return Err(Error::InvalidOperation(
                "map on a CpuAccess::None D3D11 texture tensor; allocate with Read, Write or \
                 ReadWrite"
                    .into(),
            ));
        }
        if access.writes() && !self.access.writes() {
            return Err(Error::InvalidOperation(
                "writable map on a read-only D3D11 texture tensor".into(),
            ));
        }
        Ok(())
    }

    /// Bytes of one *image* row: what this tensor's shape counts per row.
    ///
    /// The texture row for every format except a semi-planar one, whose
    /// texture is as wide as the row stride (see
    /// [`widen_semi_planar_to_pitch`]) while its shape still counts `width`
    /// samples per row. Row counts and image footprints come from here; the
    /// spacing between rows in the backing comes from
    /// [`backing_pitch`](Self::backing_pitch).
    fn image_row_bytes(&self) -> usize {
        if self.format.layout() == PixelLayout::SemiPlanar {
            self.width * self.dtype.size()
        } else {
            self.layout.tight_row_bytes()
        }
    }

    /// Row pitch of the CPU-side backing: the driver's staging pitch when
    /// there is staging, the texture's own row otherwise. Those are the same
    /// number for a semi-planar texture, which is allocated at the pitch.
    fn backing_pitch(&self) -> usize {
        self.staging
            .as_ref()
            .map_or(self.layout.tight_row_bytes(), |s| s.row_pitch)
    }

    /// Bytes addressable from a map or a pin of this backing, padding
    /// included. Offsets, view bounds and `capacity_bytes` are all in this
    /// space, because `Tensor::row_stride()` reports the same pitch and
    /// callers compute offsets from it.
    fn backing_bytes(&self) -> usize {
        self.backing_pitch() * self.layout.texture_height
    }

    /// The persistent host buffer, allocated on first use and shared by every
    /// view of this tensor.
    fn host_buffer(&self) -> &HostBuffer {
        self.host
            .get_or_init(|| HostBuffer::new(self.backing_pitch(), self.layout.texture_height))
    }

    /// Orders the reads queued after this call behind the GPU write
    /// `set_gpu_write` recorded.
    ///
    /// A staging refresh is a `CopyResource` on the immediate context, and
    /// nothing else orders it behind the context that produced the texture's
    /// contents: an unfenced convert followed by a map read whatever the
    /// texture happened to hold, which is how a destination came back blank.
    /// The recorded value is a point on this device's own fence, so a
    /// GPU-side `Wait` on it is what makes the copy see the write.
    ///
    /// Cheap by construction: a value the fence has already passed is skipped
    /// outright, and a `Wait` the driver can retire immediately is a no-op on
    /// the queue. A device that cannot wait, or a tensor with nothing
    /// recorded, does nothing -- the same degraded behaviour `order_behind`
    /// accepts, since there is no local timeline to order against.
    ///
    /// # The value has to be reachable
    ///
    /// A `Wait` queued on a value the fence never reaches does not stall one
    /// map: it stalls the immediate context for the rest of the process, and
    /// every `Signal` and every other tensor's work queued after it with it.
    /// One unreachable value would take the device down. Two rules bound it:
    ///
    /// * The value must be at or below [`D3d11Device::last_signalled`], the
    ///   newest value the counter has allocated. Every legitimate value comes
    ///   from `signal`, `signal_deferred`, or the re-signal `order_behind`
    ///   takes on import, so it is always at or below that word. Anything
    ///   above it was invented by a caller -- `ef_tensor_set_gpu_write` takes
    ///   any `u64` -- and names a point nothing will ever signal.
    /// * The device's fence must be the shared one
    ///   ([`D3d11Device::fence_is_shared`]). A copy that adopted the
    ///   published device but could not open its fence signals a private
    ///   fence from a private counter, so a value another copy recorded names
    ///   a point this fence does not have and the counter cannot tell the two
    ///   apart. That copy keeps exactly the behaviour it had before this wait
    ///   existed: a map that is unordered, but never one that hangs.
    fn wait_for_recorded_write(&self) {
        let value = self.last_gpu_write();
        if value == 0 {
            return;
        }
        if !self.device.fence_is_shared() {
            log::warn!(
                "D3D11 map: this copy signals a fence of its own, so the recorded write \
                 {value} names a point it may never reach; the staging refresh is not \
                 ordered behind it"
            );
            return;
        }
        let allocated = self.device.last_signalled();
        if value > allocated {
            log::warn!(
                "D3D11 map: the recorded write {value} is above the newest value this \
                 device's counter has allocated ({allocated}), so nothing will signal it; \
                 the wait is skipped and the staging refresh is not ordered behind it"
            );
            return;
        }
        if value <= self.device.completed_value() {
            return;
        }
        let (Some(fence), Some(ctx4)) = (self.device.fence(), self.device.ctx4()) else {
            return;
        };
        // SAFETY: both interfaces are live for the process's lifetime, and
        // the value names a point on that fence's own timeline.
        if let Err(e) = unsafe { ctx4.Wait(fence, value) } {
            log::warn!(
                "D3D11 map: queuing a wait on the recorded write {value} failed ({e}); \
                 the staging refresh may read the texture while the GPU still writes it"
            );
        }
    }

    /// Refreshes the pinned host buffer from the texture, through the staging
    /// copy. A tensor nobody has pinned has no host bytes to make coherent,
    /// and a tensor with no staging cannot read the texture back at all, so
    /// both are no-ops rather than errors.
    pub(crate) fn sync_for_cpu(&self, access: CpuAccess) -> Result<()> {
        if !access.reads() {
            return Ok(());
        }
        let (Some(buffer), Some(st)) = (self.host.get(), self.staging.as_ref()) else {
            return Ok(());
        };
        let mut state = st.state();
        if state.writer || state.readers > 0 {
            return Err(Error::InvalidOperation(
                "sync_for_cpu on a D3D11 texture tensor that is already mapped; drop the map first"
                    .into(),
            ));
        }
        self.wait_for_recorded_write();
        // SAFETY: both textures are live and identically described.
        unsafe { self.device.ctx().CopyResource(&st.tex, &self.texture) };
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        // SAFETY: `st.tex` is live, unmapped (checked above), and `mapped` is
        // a valid out-parameter.
        hr("ID3D11DeviceContext::Map (host buffer refresh)", unsafe {
            self.device
                .ctx()
                .Map(&st.tex, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
        })?;
        state.refresh_pending = false;
        let src_pitch = mapped.RowPitch as usize;
        let (base, _) = buffer.span();
        let span = src_pitch.min(buffer.row_pitch);
        for row in 0..self.layout.texture_height {
            // SAFETY: the mapping spans `rows * src_pitch` bytes and the
            // buffer `rows * row_pitch`; `span` is within both.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (mapped.pData as *const u8).add(row * src_pitch),
                    base.add(row * buffer.row_pitch),
                    span,
                );
            }
        }
        // SAFETY: `st.tex` is the texture mapped above.
        unsafe { self.device.ctx().Unmap(&st.tex, 0) };
        Ok(())
    }

    /// Publishes CPU writes to the texture from the backing that carries
    /// them: the host buffer once a writable pin or map has taken a window on
    /// it, otherwise the staging copy. Nothing is submitted when the tensor
    /// has neither, because nothing was queued.
    ///
    /// Refuses while any map of either backing is live, as
    /// [`sync_for_cpu`](Self::sync_for_cpu) does.
    pub(crate) fn sync_for_device(&self, access: CpuAccess) -> Result<()> {
        if !access.writes() {
            return Ok(());
        }
        // The same refusal `sync_for_cpu` makes, for the same reason: a live
        // map is a window onto the backing this reads from, and D3D11 rejects
        // a `CopyResource` into a mapped staging texture outright.
        //
        // Both guards are bound for the rest of the function rather than
        // dropped at the end of their check: a `map_write` on another thread
        // could otherwise take the claim between the check and the copy below,
        // and the copy would then run against a staging texture that is mapped
        // by the time it executes. Same lock order as `establish_map` -- the
        // map state first, the device's own section (inside `CopyResource` /
        // `UpdateSubresource`) second -- so this cannot invert against it.
        let _staging_claim = match self.staging.as_ref() {
            Some(st) => {
                let state = st.state();
                if state.writer || state.readers > 0 {
                    return Err(Error::InvalidOperation(
                        "sync_for_device on a D3D11 texture tensor whose staging copy is \
                         mapped; drop the map first"
                            .into(),
                    ));
                }
                Some(state)
            }
            None => None,
        };
        let _host_claim = match self.host.get() {
            Some(buffer) => {
                let state = buffer.state();
                if state.writer || state.readers > 0 {
                    return Err(Error::InvalidOperation(
                        "sync_for_device on a D3D11 texture tensor whose host buffer is \
                         mapped; drop the map first"
                            .into(),
                    ));
                }
                Some(state)
            }
            None => None,
        };
        // Whichever backing the caller actually wrote through. The host buffer
        // only wins once someone has taken a writable window on it: a tensor
        // that was merely pinned for reading would otherwise publish the
        // buffer's untouched (zero) rows over the texture.
        //
        // The flag is sticky -- one writable pin or map sets it for the
        // tensor's life, and a later staging write does not clear it. That is
        // the conservative direction and it is what a pin means: a pin outlives
        // every guard, so the holder may write through it at any time, and
        // there is no drop to observe the last write at. A caller that mixes
        // both backings on one tensor owns the ordering between them, which is
        // the coherency obligation `crate::pin` already states.
        match (self.wrote_host.load(Ordering::Acquire), self.host.get()) {
            (true, Some(buffer)) => upload_buffer(self.device, &self.texture, buffer),
            _ => match &self.staging {
                // SAFETY: both textures are live and identically described,
                // and `_staging_claim` is held across this call, so the
                // staging texture cannot be mapped while the copy is issued.
                Some(st) => unsafe { self.device.ctx().CopyResource(&self.texture, &st.tex) },
                None => return Ok(()),
            },
        }
        publish(self.device);
        Ok(())
    }

    /// A stable host address for this tensor's rows, valid until the pin
    /// drops and across any number of maps and frames.
    ///
    /// Deliberately not the staging mapping: a pin outlives every guard, so
    /// mapping the staging texture here would keep it mapped forever, block
    /// every later map and freeze the bytes at the moment of the pin. The
    /// address is the storage's own host buffer instead; `sync_for_cpu` and
    /// `sync_for_device` move bytes between it and the texture.
    pub(crate) fn host_pin<'a>(&self, access: CpuAccess) -> Result<crate::pin::HostPin<'a>> {
        self.require_access(self.access)?;
        if access.writes() {
            // A pin outlives every guard, so there is no drop to notice the
            // write at: `sync_for_device` reads this to decide that the host
            // buffer, not the staging copy, is what the caller wrote through.
            self.wrote_host.store(true, Ordering::Release);
        }
        let (base, len) = self.host_buffer().span();
        // SAFETY: `view` bounds-checks `view_offset` against `backing_bytes()`,
        // which is the length this buffer was allocated at.
        let at = unsafe { base.add(self.view_offset) };
        // The cell, not the buffer: holding it is what keeps the address live.
        let keepalive: Arc<dyn Send + Sync> = self.host.clone();
        Ok(crate::pin::HostPin::new(
            keepalive,
            at,
            len.saturating_sub(self.view_offset),
        ))
    }

    /// One CPU access window, released when the pin drops. Readable tensors
    /// go through the staging texture; a write-only one writes the host
    /// buffer and uploads it.
    fn pin_impl<'a>(
        &self,
        access: CpuAccess,
        non_blocking: bool,
    ) -> Result<crate::pin::HostPin<'a>> {
        self.require_access(access)?;
        self.refuse_partial_write(access)?;
        match self.staging.as_ref() {
            Some(st) => self.staging_pin(st, access, non_blocking),
            None => self.buffer_pin(access),
        }
    }

    /// Refuses a write-only window that covers less than the whole backing.
    ///
    /// Unmapping a writable window publishes the *whole* backing -- a
    /// `CopyResource` of the staging texture, or an `UpdateSubresource` of the
    /// host buffer -- and a write-only window is not refreshed from the
    /// texture first. For a whole-tensor map that is the documented contract
    /// ("bytes the caller leaves untouched are undefined"). For a sub-view it
    /// is not a contract the caller can meet: the window spans its own rows
    /// only, so the rows outside it get published as undefined bytes on first
    /// use and as a stale snapshot afterwards. `ReadWrite` is the same window
    /// with the refresh, so it says what it costs.
    fn refuse_partial_write(&self, access: CpuAccess) -> Result<()> {
        if access != CpuAccess::Write {
            return Ok(());
        }
        let backing = self.backing_bytes();
        if self.view_offset == 0 && self.pitched_extent() >= backing {
            return Ok(());
        }
        Err(Error::InvalidArgument(format!(
            "write-only map of a D3D11 texture tensor window ({} bytes at offset {}) that is \
             shorter than the {backing}-byte backing: unmapping publishes the whole texture, \
             so the rows outside the window would become undefined; map the window with \
             CpuAccess::ReadWrite instead",
            self.pitched_extent(),
            self.view_offset
        )))
    }

    fn staging_pin<'a>(
        &self,
        st: &Arc<Staging>,
        access: CpuAccess,
        non_blocking: bool,
    ) -> Result<crate::pin::HostPin<'a>> {
        let writes = access.writes();
        // The whole claim-and-map runs under one lock so a second reader
        // cannot observe a claimed-but-not-yet-mapped `base`. No guard exists
        // inside this scope: dropping one here would re-enter the mutex.
        let base = {
            let mut state = st.state();
            match state.claim(writes)? {
                // Joined a live read mapping: same snapshot, one more reader,
                // no second `Map` and no second copy into the staging texture.
                Some(base) => base,
                None => self.establish_map(st, &mut state, access, non_blocking)?,
            }
        };
        let guard = StagingGuard {
            device: self.device,
            texture: self.texture.clone(),
            staging: Arc::clone(st),
            writer: writes,
        };
        Ok(self.pin_over(base as *mut u8, st.bytes(), guard))
    }

    /// Refreshes the staging copy if needed and maps it, with the claim
    /// already taken. Releases the claim on any failure.
    fn establish_map(
        &self,
        st: &Staging,
        state: &mut MapState,
        access: CpuAccess,
        non_blocking: bool,
    ) -> Result<usize> {
        // `refresh_pending` is a non-blocking optimisation, not a cache: it
        // exists so a caller polling `try_map` does not queue one
        // `CopyResource` per attempt. A blocking map has to refresh whatever
        // is pending, because the copy still queued was issued before whatever
        // the GPU has written since -- a poll that gave up followed by a
        // `map()` would otherwise hand back the previous frame. The blocking
        // `Map` waits for what it queues, so the extra copy costs nothing.
        if access.reads() && !(non_blocking && state.refresh_pending) {
            self.wait_for_recorded_write();
            // SAFETY: both textures are live, identically described, and the
            // staging texture is unmapped (this caller holds the only claim).
            unsafe { self.device.ctx().CopyResource(&st.tex, &self.texture) };
            state.refresh_pending = true;
            if non_blocking {
                // A non-blocking map can only succeed once the copy has been
                // submitted; without this a polling caller spins forever
                // while the queue grows.
                publish(self.device);
            }
        }
        let mode = if access.writes() {
            D3D11_MAP_READ_WRITE
        } else {
            D3D11_MAP_READ
        };
        let flags = if non_blocking {
            D3D11_MAP_FLAG_DO_NOT_WAIT.0 as u32
        } else {
            0
        };
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        // SAFETY: `st.tex` is live and `mapped` is a valid out-parameter.
        let result = unsafe {
            self.device
                .ctx()
                .Map(&st.tex, 0, mode, flags, Some(&mut mapped))
        };
        if let Err(e) = result {
            state.release(access.writes());
            if e.code() == DXGI_ERROR_WAS_STILL_DRAWING {
                // `refresh_pending` stays set, so a retry re-attempts only the
                // map rather than queueing a second copy.
                return Err(Error::IoError(std::io::Error::from(
                    std::io::ErrorKind::WouldBlock,
                )));
            }
            return hr("ID3D11DeviceContext::Map", Err(e));
        }
        // The map consumed the queued copy.
        state.refresh_pending = false;
        state.base = mapped.pData as usize;
        Ok(state.base)
    }

    fn buffer_pin<'a>(&self, access: CpuAccess) -> Result<crate::pin::HostPin<'a>> {
        let buffer = self.host_buffer();
        buffer.state().claim(access.writes())?;
        if access.writes() {
            // See `host_pin`: this is what tells `sync_for_device` which
            // backing carries the caller's writes.
            self.wrote_host.store(true, Ordering::Release);
        }
        let (base, len) = buffer.span();
        let guard = HostBufferGuard {
            device: self.device,
            texture: self.texture.clone(),
            host: Arc::clone(&self.host),
            // A read map of a write-only tensor must not push its host bytes
            // into the texture; only a writable map publishes.
            writeback: access.writes(),
        };
        Ok(self.pin_over(base, len, guard))
    }

    /// Offsets `base` by this tensor's view offset and wraps it in a pin
    /// whose keepalive is `guard`.
    fn pin_over<'a, G: Send + Sync + 'static>(
        &self,
        base: *mut u8,
        len: usize,
        guard: G,
    ) -> crate::pin::HostPin<'a> {
        // SAFETY: `view` bounds-checks `view_offset` against `backing_bytes()`,
        // which is the length of both backings.
        let at = unsafe { base.add(self.view_offset) };
        crate::pin::HostPin::new(Arc::new(guard), at, len.saturating_sub(self.view_offset))
    }

    pub(crate) fn scoped_pin<'a>(&self, access: CpuAccess) -> Result<crate::pin::HostPin<'a>> {
        self.pin_impl(access, false)
    }

    /// `scoped_pin` that reports `WouldBlock` instead of stalling while the
    /// GPU still holds the staging copy.
    pub(crate) fn try_scoped_pin<'a>(&self, access: CpuAccess) -> Result<crate::pin::HostPin<'a>> {
        self.pin_impl(access, true)
    }

    /// Bytes an ordinary map exposes: this tensor's own rows at the backing
    /// pitch.
    ///
    /// Row padding is addressable -- `Tensor::row_stride()` reports the same
    /// pitch and callers iterate rows by it -- so the extent is pitched, not
    /// tight. What it is *not* is everything left in the parent allocation: a
    /// sub-view covers its own rows, which is what a batched destination's
    /// per-band map depends on to match the same convert done standalone.
    ///
    /// The granularity is a whole pitched row, because the pitch is the only
    /// row spacing the backing has: a shape covering part of a row still maps
    /// to the end of that row, since rounding down would hand back a window
    /// the caller cannot index by stride.
    ///
    /// Rows come from the shape's own byte count over the image's tight row,
    /// so one expression covers packed, planar and semi-planar shapes: each
    /// one's product over `tight_row_bytes` is its row count in the texture.
    /// The byte count uses `dtype`, not `size_of::<T>()`: the shape counts
    /// samples of the image's element type, while `T` is only the type a map
    /// hands back, and the two differ when a caller maps an F16 image as bytes.
    fn pitched_extent(&self) -> usize {
        let tight_row = self.image_row_bytes();
        if tight_row == 0 {
            return 0;
        }
        let tight = self.shape.iter().product::<usize>() * self.dtype.size();
        tight.div_ceil(tight_row) * self.backing_pitch()
    }

    /// Wrap a pin as a view of this tensor's shape. `byte_size_override` is
    /// `None` for an ordinary map, which exposes
    /// [`pitched_extent`](Self::pitched_extent); the padded extent of a
    /// differently-strided window comes through
    /// [`map_with_byte_size`](Self::map_with_byte_size), which
    /// `Tensor::map_with` calls when the tensor carries a row stride.
    fn view_of<'a>(
        &self,
        pin: crate::pin::HostPin<'a>,
        access: CpuAccess,
        byte_size_override: Option<usize>,
    ) -> crate::view::HostView<'a, T>
    where
        T: 'a,
    {
        let bytes = byte_size_override.unwrap_or_else(|| self.pitched_extent());
        crate::view::HostView::new(pin, self.shape.clone(), Some(bytes), access)
    }

    /// Bounds-check a padded-extent request against this tensor's window.
    fn check_byte_size(&self, bytes: usize) -> Result<()> {
        let capacity = self.backing_bytes().saturating_sub(self.view_offset);
        if bytes > capacity {
            return Err(Error::InsufficientCapacity {
                needed: bytes,
                capacity,
            });
        }
        Ok(())
    }

    /// Map exposing `bytes` of the pitched backing rather than the shape's
    /// tight extent -- the padded-row path, mirroring
    /// `IoSurfaceTensor::map_with_byte_size`.
    pub(crate) fn map_with_byte_size<'a>(
        &self,
        bytes: usize,
        access: CpuAccess,
    ) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        self.check_byte_size(bytes)?;
        Ok(self.view_of(self.scoped_pin(access)?, access, Some(bytes)))
    }

    /// [`map_with_byte_size`](Self::map_with_byte_size) that reports
    /// `WouldBlock` instead of stalling on the staging refresh.
    pub(crate) fn try_map_with_byte_size<'a>(
        &self,
        bytes: usize,
        access: CpuAccess,
    ) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        self.check_byte_size(bytes)?;
        Ok(self.view_of(self.try_scoped_pin(access)?, access, Some(bytes)))
    }

    pub(crate) fn texture_ptr(&self) -> *mut c_void {
        self.texture.as_raw()
    }

    pub(crate) fn layout(&self) -> D3d11ImageLayout {
        self.layout
    }

    // The image crate reads this when the tensor is a render destination; no
    // accessor exposes it yet.
    #[allow(dead_code)]
    pub(crate) fn is_render_target(&self) -> bool {
        self.desc_bind_flags & D3D11_BIND_RENDER_TARGET.0 as u32 != 0
    }

    /// A duplicate of the texture's shared NT handle, owned by the caller.
    pub(crate) fn shared_handle(&self) -> Result<OwnedHandle> {
        duplicate_handle(&self.nt_handle).map_err(Error::IoError)
    }

    /// The tensor's own NT handle value, valid while the tensor lives. For
    /// descriptors and the D1 blob, which carry a value rather than ownership.
    pub(crate) fn shared_handle_value(&self) -> usize {
        self.nt_handle.as_raw_handle() as usize
    }

    /// The fence point covering the newest GPU write recorded on this tensor,
    /// with a duplicate of the fence handle for a consumer to wait on.
    pub(crate) fn gpu_completion(&self) -> Result<Option<GpuCompletion>> {
        let value = self.last_gpu_write();
        if value == 0 {
            return Ok(None);
        }
        Ok(Some(GpuCompletion {
            fence: self.device.fence_shared_handle().map_err(Error::IoError)?,
            value,
        }))
    }

    /// Records a fence value a producer signalled after writing this texture.
    /// Monotonic: an older value never displaces a newer one.
    ///
    /// Any `u64` is accepted, because the C and Python surfaces take one from
    /// a caller and a producer's value is not this crate's to validate. What
    /// a value above the device's own counter does *not* do is reach the
    /// immediate context: `wait_for_recorded_write` ignores it rather than
    /// queue a `Wait` nothing will ever satisfy. It is still reported by
    /// [`gpu_completion`](Self::gpu_completion), which is a statement about
    /// the producer's timeline, not this device's.
    pub(crate) fn set_gpu_write(&self, value: u64) {
        self.last_gpu_write.fetch_max(value, Ordering::AcqRel);
    }

    pub(crate) fn last_gpu_write(&self) -> u64 {
        self.last_gpu_write.load(Ordering::Acquire)
    }

    /// The row pitch a CPU map or pin sees, or `None` when there is none to
    /// report. One number for both, which is why the host buffer is allocated
    /// at this pitch too.
    ///
    /// The staging pitch when there is staging: that is what a map is
    /// literally laid out at, whatever the layout table says.
    ///
    /// A semi-planar tensor answers even without staging, because its texture
    /// is as wide as the driver's row pitch by construction -- `new_image`
    /// widens its own allocations to it (see [`widen_semi_planar_to_pitch`])
    /// and [`validate_external`] refuses a wrapped texture that is not already
    /// as wide as its own pitch -- so the two branches are the same number and
    /// the write-only host buffer is laid out at it too. Reporting `None`
    /// there would tell a producer to write chroma lines at the tight pitch
    /// into a wider texture, which is the mismatch the widening exists to
    /// remove.
    ///
    /// Every other format is `None` without staging: a write-only tensor's
    /// host buffer and a hardware-only tensor both have tight rows or no rows
    /// at all.
    pub(crate) fn image_backing_row_stride(&self) -> Option<usize> {
        if let Some(st) = self.staging.as_ref() {
            return Some(st.row_pitch);
        }
        // R8, so the texture's row in bytes is its width in texels.
        (self.format.layout() == PixelLayout::SemiPlanar).then(|| self.layout.tight_row_bytes())
    }

    /// Bytes the image occupies with no row padding. The logical extent
    /// `set_logical_shape` measures against, not the addressable one -- so the
    /// image's rows, not the texture's, which differ when a semi-planar
    /// texture was widened to the driver's pitch.
    pub(crate) fn tight_bytes(&self) -> usize {
        self.image_row_bytes() * self.layout.texture_height
    }

    /// Imports this texture into CUDA, for `Tensor::try_init_d3d11_cuda` to
    /// attach at allocation and at both wrap constructors.
    ///
    /// Best-effort, like the DMA-BUF import on Linux: `None` when no CUDA
    /// runtime loaded, when the adapter has no CUDA device (WARP), or when the
    /// driver refuses the import. The tensor is then a perfectly ordinary
    /// texture tensor whose `cuda_map()` is `None`.
    ///
    /// The texture's own NT handle is *borrowed* for the call -- CUDA
    /// duplicates a Win32 handle it imports -- while the fence's is a fresh
    /// duplicate the CUDA handle takes over, because it outlives the call.
    pub(crate) fn try_init_cuda(&self) -> Option<crate::cuda::CudaHandle> {
        if !crate::cuda::is_cuda_available() {
            return None;
        }
        // No fence means no GPU-side ordering, which the import tolerates: it
        // then copies without waiting, as it does for a tensor no producer has
        // recorded a write on.
        let fence = self.device.fence_shared_handle().ok();
        crate::cuda::import_d3d11_texture(
            self.nt_handle.as_raw_handle(),
            self.layout,
            self.device.adapter_ptr(),
            fence,
            Arc::clone(&self.last_gpu_write),
        )
    }
}

impl<T> fmt::Debug for D3d11TextureTensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("D3d11TextureTensor")
            .field("name", &self.name)
            .field("shape", &self.shape)
            .field("layout", &self.layout)
            .field("access", &self.access)
            .field("texture", &self.texture.as_raw())
            .finish()
    }
}

/// Submits the immediate context, so a copy that has just published CPU
/// writes reaches a device that opened this texture's shared handle. D3D11
/// does not submit a shared resource's writes on its own, and these textures
/// carry no keyed mutex, so without this another device reads stale bytes.
///
/// This submits the whole immediate context, not just the copy -- including
/// commands the GL worker has in flight on the same device. That is sound
/// rather than merely tolerable: the context is used under
/// `ID3D11Multithread` protection, and an early submit changes when work
/// runs, never what it computes.
fn publish(device: &D3d11Device) {
    // SAFETY: the immediate context is live for the process's lifetime.
    unsafe { device.ctx().Flush() };
}

/// Uploads the host buffer's rows into the texture at the pitch the buffer
/// was allocated with.
fn upload_buffer(device: &D3d11Device, texture: &ID3D11Texture2D, buffer: &HostBuffer) {
    let (base, _) = buffer.span();
    // SAFETY: the buffer is live for the call, the destination texture is
    // live, and `row_pitch` is the pitch its rows are laid out at.
    unsafe {
        device
            .ctx()
            .UpdateSubresource(texture, 0, None, base.cast(), buffer.row_pitch as u32, 0);
    }
}

/// Releases one claim on the staging mapping, unmapping when it was the last
/// and, for the writer, publishing the rows to the texture.
struct StagingGuard {
    device: &'static D3d11Device,
    texture: ID3D11Texture2D,
    staging: Arc<Staging>,
    writer: bool,
}

impl Drop for StagingGuard {
    fn drop(&mut self) {
        let mut state = self.staging.state();
        if !state.release(self.writer) {
            // Another reader still holds the mapping.
            return;
        }
        // SAFETY: the staging texture is live and mapped by this guard.
        unsafe { self.device.ctx().Unmap(&self.staging.tex, 0) };
        if self.writer {
            // SAFETY: both textures are live and identically described.
            unsafe {
                self.device
                    .ctx()
                    .CopyResource(&self.texture, &self.staging.tex)
            };
            publish(self.device);
        }
    }
}

/// Releases one claim on the host buffer and, for a writable map, uploads it.
struct HostBufferGuard {
    device: &'static D3d11Device,
    texture: ID3D11Texture2D,
    host: Arc<OnceLock<HostBuffer>>,
    writeback: bool,
}

impl Drop for HostBufferGuard {
    fn drop(&mut self) {
        let buffer = self
            .host
            .get()
            .expect("the guard was built from a live host buffer");
        // The claim is held across the upload, as `StagingGuard::drop` holds
        // its lock across the equivalent copy: releasing first would let
        // another thread take a write map and mutate the rows in the middle of
        // `UpdateSubresource`.
        let mut state = buffer.state();
        if self.writeback {
            upload_buffer(self.device, &self.texture, buffer);
            publish(self.device);
        }
        state.release(self.writeback);
    }
}

impl<T> TensorTrait<T> for D3d11TextureTensor<T>
where
    T: Num + Clone + fmt::Debug + Send + Sync,
{
    fn new(_shape: &[usize], _name: Option<&str>) -> Result<Self> {
        Err(Error::NotImplemented(
            "D3D11 texture tensors are image-formatted; use Tensor::image with \
             TensorMemory::DmaBuf"
                .to_owned(),
        ))
    }

    fn memory(&self) -> TensorMemory {
        // Unified variant: Windows reports DmaBuf, as macOS and Linux do. The
        // variant name is shared; the inner storage type differs per platform.
        TensorMemory::DmaBuf
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn reshape(&mut self, shape: &[usize]) -> Result<()> {
        let new_elems: usize = shape.iter().product();
        let cur_elems: usize = self.shape.iter().product();
        if new_elems != cur_elems {
            return Err(Error::InvalidShape(format!(
                "reshape: element count mismatch ({cur_elems} -> {new_elems})"
            )));
        }
        self.shape = shape.to_vec();
        Ok(())
    }

    fn map_with<'a>(&self, access: CpuAccess) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        let _span = tracing::trace_span!("tensor.map", memory = "d3d11_texture", ?access).entered();
        Ok(self.view_of(self.scoped_pin(access)?, access, None))
    }

    /// The one backing whose `try_map_with` is not `map_with`: mapping the
    /// staging texture waits for the refresh copy, so a non-blocking caller is
    /// told to come back rather than stalled.
    ///
    /// No span of its own: `Tensor::map_impl` already opens one `tensor.map`
    /// span carrying `non_blocking`, and that is the single spelling of the
    /// event.
    fn try_map_with<'a>(&self, access: CpuAccess) -> Result<crate::view::HostView<'a, T>>
    where
        T: 'a,
    {
        Ok(self.view_of(self.try_scoped_pin(access)?, access, None))
    }

    fn buffer_identity(&self) -> &BufferIdentity {
        &self.identity
    }

    /// Addressable bytes from this tensor's own window, row padding included:
    /// the space `view` offsets are measured in.
    fn capacity_bytes(&self) -> usize {
        self.backing_bytes().saturating_sub(self.view_offset)
    }

    fn set_logical_shape(&mut self, shape: &[usize]) -> Result<()> {
        if shape.is_empty() {
            return Err(Error::InvalidSize(0));
        }
        // The image's own size, not the padded one: a logical shape describes
        // pixels, and padding is not addressable as data.
        let needed = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        let capacity = self.tight_bytes();
        if needed > capacity {
            return Err(Error::InsufficientCapacity { needed, capacity });
        }
        self.shape = shape.to_vec();
        Ok(())
    }

    /// Zero-copy sub-region view sharing this texture, its staging, its host
    /// buffer, its NT handle and its [`BufferIdentity`], positioned at
    /// `offset_bytes` from this tensor's own window.
    ///
    /// `offset_bytes` is in backing (pitched) space, matching what
    /// `Tensor::row_stride()` reports, so a caller offsets by whole rows with
    /// `n * row_stride`.
    fn view(&self, offset_bytes: usize, shape: &[usize]) -> Result<Self> {
        if !offset_bytes.is_multiple_of(std::mem::align_of::<T>()) {
            return Err(Error::InvalidOperation(format!(
                "D3d11TextureTensor::view: offset {offset_bytes} not aligned to \
                 align_of::<T>()={}",
                std::mem::align_of::<T>()
            )));
        }
        let abs_offset = self
            .view_offset
            .checked_add(offset_bytes)
            .ok_or(Error::InvalidSize(offset_bytes))?;
        let capacity = self.backing_bytes();
        if abs_offset > capacity {
            return Err(Error::InsufficientCapacity {
                needed: abs_offset,
                capacity,
            });
        }
        let logical = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        let needed = abs_offset
            .checked_add(logical)
            .ok_or(Error::InvalidSize(logical))?;
        if needed > capacity {
            return Err(Error::InsufficientCapacity { needed, capacity });
        }
        Ok(Self {
            device: self.device,
            texture: self.texture.clone(),
            layout: self.layout,
            desc_bind_flags: self.desc_bind_flags,
            width: self.width,
            height: self.height,
            format: self.format,
            dtype: self.dtype,
            shape: shape.to_vec(),
            name: self.name.clone(),
            access: self.access,
            staging: self.staging.clone(),
            host: Arc::clone(&self.host),
            nt_handle: Arc::clone(&self.nt_handle),
            identity: self.identity.clone(),
            last_gpu_write: Arc::clone(&self.last_gpu_write),
            imported_fence: self.imported_fence.clone(),
            wrote_host: Arc::clone(&self.wrote_host),
            view_offset: abs_offset,
            _marker: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests_support {
    use std::os::windows::io::{FromRawHandle, OwnedHandle};
    use windows::core::{Interface, PCWSTR};
    use windows::Win32::Foundation::{GENERIC_ALL, HANDLE, HMODULE};
    use windows::Win32::Graphics::Direct3D::{D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_WARP};
    use windows::Win32::Graphics::Direct3D11::{
        D3D11CreateDevice, ID3D11Device, ID3D11Device1, ID3D11Device5, ID3D11DeviceContext,
        ID3D11DeviceContext4, ID3D11Fence, ID3D11Texture2D, D3D11_CPU_ACCESS_READ,
        D3D11_CREATE_DEVICE_FLAG, D3D11_FENCE_FLAG_SHARED, D3D11_MAPPED_SUBRESOURCE,
        D3D11_MAP_READ, D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC, D3D11_USAGE_STAGING,
    };
    use windows::Win32::Graphics::Dxgi::IDXGIAdapter;

    /// A device other than the process device, standing in for a consumer in
    /// another API that opens the tensor's shared handle.
    pub(super) struct SecondDevice {
        pub dev: ID3D11Device,
        pub dev1: ID3D11Device1,
        pub ctx: ID3D11DeviceContext,
    }

    /// A second D3D11 device on the same adapter kind as the process device.
    pub(super) fn second_device() -> SecondDevice {
        let driver = if super::device().unwrap().is_warp() {
            D3D_DRIVER_TYPE_WARP
        } else {
            D3D_DRIVER_TYPE_HARDWARE
        };
        let mut dev: Option<ID3D11Device> = None;
        let mut ctx: Option<ID3D11DeviceContext> = None;
        // SAFETY: documented creation call; both out-parameters are valid locals.
        unsafe {
            D3D11CreateDevice(
                None::<&IDXGIAdapter>,
                driver,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_FLAG(0),
                None,
                D3D11_SDK_VERSION,
                Some(&mut dev),
                None,
                Some(&mut ctx),
            )
        }
        .expect("second D3D11 device");
        let dev = dev.expect("D3D11CreateDevice returned a device");
        let dev1 = dev.cast::<ID3D11Device1>().expect("ID3D11Device1");
        SecondDevice {
            dev,
            dev1,
            ctx: ctx.expect("D3D11CreateDevice returned a context"),
        }
    }

    /// Opens `handle` on `other`, copies it to a staging texture of `other`'s
    /// and returns the tight `w * h * bpp` bytes.
    pub(super) fn read_through(
        other: &SecondDevice,
        handle: HANDLE,
        w: usize,
        h: usize,
        bpp: usize,
    ) -> Vec<u8> {
        // SAFETY: `handle` is a shared NT handle valid in this process.
        let tex: ID3D11Texture2D = unsafe { other.dev1.OpenSharedResource1(handle) }
            .expect("ID3D11Device1::OpenSharedResource1");
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        // SAFETY: `tex` is live and `desc` is a valid out-parameter.
        unsafe { tex.GetDesc(&mut desc) };
        desc.Usage = D3D11_USAGE_STAGING;
        desc.BindFlags = 0;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ.0 as u32;
        desc.MiscFlags = 0;
        let mut staging: Option<ID3D11Texture2D> = None;
        // SAFETY: `desc` is fully initialised and `staging` is a valid out-parameter.
        unsafe { other.dev.CreateTexture2D(&desc, None, Some(&mut staging)) }
            .expect("staging texture on the second device");
        let staging = staging.expect("CreateTexture2D returned a texture");
        // SAFETY: both textures are live on `other` and identically described.
        unsafe { other.ctx.CopyResource(&staging, &tex) };
        let mut m = D3D11_MAPPED_SUBRESOURCE::default();
        // SAFETY: `staging` is live and `m` is a valid out-parameter.
        unsafe { other.ctx.Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut m)) }
            .expect("map the staging copy");
        let row = w * bpp;
        let mut out = vec![0u8; row * h];
        for r in 0..h {
            // SAFETY: the mapping spans `h * RowPitch` bytes and `row <= RowPitch`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (m.pData as *const u8).add(r * m.RowPitch as usize),
                    out.as_mut_ptr().add(r * row),
                    row,
                );
            }
        }
        // SAFETY: `staging` is the texture this function mapped above.
        unsafe { other.ctx.Unmap(&staging, 0) };
        out
    }

    /// A shared fence created on `other`, with the NT handle a consumer in
    /// this process opens to wait on it.
    ///
    /// The producer side of the import path: a second device signalling a
    /// fence of its own is what a real producer in another API or another
    /// process does, and the only way in one process to put GPU work on a
    /// queue this process's device does not already order itself against.
    pub(super) fn shared_fence(other: &SecondDevice) -> (ID3D11Fence, OwnedHandle) {
        let d5 = other.dev.cast::<ID3D11Device5>().expect("ID3D11Device5");
        let mut f: Option<ID3D11Fence> = None;
        // SAFETY: `d5` is live and `f` is a valid out-parameter.
        unsafe { d5.CreateFence(0, D3D11_FENCE_FLAG_SHARED, &mut f) }.expect("CreateFence");
        let fence = f.expect("CreateFence returned a fence");
        // SAFETY: `fence` is live; the handle is unnamed and this process owns it.
        let handle = unsafe { fence.CreateSharedHandle(None, GENERIC_ALL.0, PCWSTR::null()) }
            .expect("ID3D11Fence::CreateSharedHandle");
        // SAFETY: an NT handle this call just created and nothing else holds.
        (fence, unsafe { OwnedHandle::from_raw_handle(handle.0) })
    }

    /// Fills the texture `handle` names with `byte` from `other`, then
    /// signals `fence` at `value` behind that write and submits.
    ///
    /// The signal is queued after the write on the same context, so the
    /// value covers it: a consumer that waits for the value is ordered
    /// behind the bytes, and one that does not is racing them.
    ///
    /// `churn` zero-filled uploads go in front of the real one so that race
    /// is one a test can lose. A single small upload retires long before a
    /// consumer gets to its own copy, and a test that cannot fail without
    /// the wait proves nothing about it; reading this texture early reads
    /// one of the zero fills instead.
    #[allow(clippy::too_many_arguments)] // one write, spelled out
    pub(super) fn write_through(
        other: &SecondDevice,
        handle: HANDLE,
        w: usize,
        h: usize,
        bpp: usize,
        byte: u8,
        churn: usize,
        fence: &ID3D11Fence,
        value: u64,
    ) {
        // SAFETY: `handle` is a shared NT handle valid in this process.
        let tex: ID3D11Texture2D = unsafe { other.dev1.OpenSharedResource1(handle) }
            .expect("ID3D11Device1::OpenSharedResource1");
        let zeros = vec![0u8; w * bpp * h];
        let rows = vec![byte; w * bpp * h];
        for source in std::iter::repeat_n(&zeros, churn).chain(std::iter::once(&rows)) {
            // SAFETY: `tex` is live on `other`, and the source spans `h` rows
            // of `w * bpp` bytes, which is the pitch passed here.
            unsafe {
                other.ctx.UpdateSubresource(
                    &tex,
                    0,
                    None,
                    source.as_ptr().cast(),
                    (w * bpp) as u32,
                    0,
                )
            };
        }
        let ctx4 = other
            .ctx
            .cast::<ID3D11DeviceContext4>()
            .expect("ID3D11DeviceContext4");
        // SAFETY: both interfaces are live and `value` is fresh on this fence.
        unsafe { ctx4.Signal(fence, value) }.expect("ID3D11DeviceContext4::Signal");
        // SAFETY: the second device's immediate context is live. Without the
        // submit the consumer waits on a value that was never sent.
        unsafe { other.ctx.Flush() };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorMapTrait;
    use std::time::{Duration, Instant};
    use windows::Win32::Foundation::WAIT_OBJECT_0;
    use windows::Win32::System::Threading::WaitForSingleObject;

    /// A `ReadWrite` RGBA8 tensor of `w` x `h`, the shape most tests want.
    fn rgba(w: usize, h: usize, access: CpuAccess) -> D3d11TextureTensor<u8> {
        let shape = PixelFormat::Rgba.allocation_shape(w, h).unwrap();
        D3d11TextureTensor::<u8>::new_image(
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            &shape,
            None,
            access,
        )
        .unwrap()
    }

    /// A `try_map` that gave up leaves one refresh copy queued and unconsumed.
    /// The next *blocking* map must re-queue rather than ride that copy: it
    /// was issued before whatever the GPU has written since, so honouring the
    /// flag there hands the caller the previous frame.
    ///
    /// The later write is a real GPU texture-to-texture copy, not a CPU
    /// staging write. A staging write publishes the whole staging texture on
    /// unmap, which refreshes the pending copy as a side effect and lets the
    /// pre-fix code pass; only a write that lands in the texture behind the
    /// staging copy's back tells the two apart. Reverting the gate in
    /// `establish_map` to `!state.refresh_pending` fails this test.
    #[test]
    fn a_blocking_map_refreshes_past_an_abandoned_try_maps_copy() {
        let d = device().unwrap();
        let old = rgba(64, 64, CpuAccess::ReadWrite);
        let new = rgba(64, 64, CpuAccess::ReadWrite);
        for (t, byte) in [(&old, 0x11u8), (&new, 0x22u8)] {
            let mut m = t.map_with(CpuAccess::Write).unwrap();
            m.as_mut_slice().fill(byte);
        }

        // Exactly the state a `try_map` that answered `WouldBlock` leaves
        // behind: one refresh copy queued into the staging texture, and the
        // flag that says a retry must not queue a second. Built directly so
        // the precondition is deterministic rather than a race with the GPU.
        {
            let st = old.staging.as_ref().unwrap();
            let mut state = st.state();
            // SAFETY: both textures are live and identically described, and
            // this claim-free scope holds the state lock, so nothing else can
            // map the staging texture while the copy is issued.
            unsafe { d.ctx().CopyResource(&st.tex, &old.texture) };
            state.refresh_pending = true;
        }

        // The GPU write the abandoned copy predates.
        // SAFETY: both textures are live and identically described.
        unsafe { d.ctx().CopyResource(&old.texture, &new.texture) };
        d.signal();

        // Only the logical bytes of each row are compared. Both write maps
        // above filled the whole pitched extent, so on an adapter that pads a
        // 64-texel RGBA8 row the staging padding still holds the old byte:
        // the GPU copy moved the texture, which has no padding, and a refresh
        // of the staging texture is not obliged to overwrite its row tails.
        let mapped = old.map_with(CpuAccess::Read).unwrap();
        let pitch = old.image_backing_row_stride().unwrap();
        let row_bytes = old.layout().tight_row_bytes();
        let bytes = mapped.as_slice();
        let logical = |r: usize| &bytes[r * pitch..r * pitch + row_bytes];
        let stale = (0..old.layout().texture_height)
            .flat_map(|r| logical(r).iter())
            .filter(|&&b| b == 0x11)
            .count();
        assert_eq!(
            stale, 0,
            "the blocking map returned {stale} logical bytes of the copy queued before the GPU write"
        );
        for r in 0..old.layout().texture_height {
            assert!(
                logical(r).iter().all(|&b| b == 0x22),
                "row {r} is not the byte the GPU write put there"
            );
        }
    }

    fn rows() -> Vec<(PixelFormat, DType, usize, usize)> {
        vec![
            (PixelFormat::Rgba, DType::U8, 640, 480),
            (PixelFormat::Bgra, DType::U8, 641, 3),
            (PixelFormat::Rgb, DType::U8, 640, 480),
            (PixelFormat::Grey, DType::U8, 641, 481),
            (PixelFormat::Nv12, DType::U8, 640, 480),
            (PixelFormat::Nv16, DType::U8, 640, 480),
            (PixelFormat::Nv24, DType::U8, 640, 480),
            (PixelFormat::Yuyv, DType::U8, 640, 480),
            (PixelFormat::PlanarRgb, DType::F16, 640, 480),
            (PixelFormat::PlanarRgb, DType::F32, 640, 480),
            (PixelFormat::Rgb, DType::F32, 640, 480),
            (PixelFormat::Rgba, DType::F16, 640, 480),
        ]
    }

    #[test]
    fn every_layout_row_allocates_and_round_trips_through_map() {
        for (fmt, dt, w, h) in rows() {
            let shape = fmt.allocation_shape(w, h).unwrap();
            let t = D3d11TextureTensor::<u8>::new_image(
                w,
                h,
                fmt,
                dt,
                &shape,
                None,
                CpuAccess::ReadWrite,
            )
            .unwrap_or_else(|e| panic!("{fmt:?}/{dt:?}: {e}"));
            let tight = t.tight_bytes();
            let pitch = t
                .image_backing_row_stride()
                .expect("mappable tensors know their pitch");
            assert!(pitch >= t.layout().tight_row_bytes());
            let pattern: Vec<u8> = (0..tight).map(|i| (i * 7 + 13) as u8).collect();
            {
                let mut v = t.map_with(CpuAccess::Write).unwrap();
                let rows = t.layout().texture_height;
                let row_bytes = t.layout().tight_row_bytes();
                let dst = v.as_mut_slice();
                for r in 0..rows {
                    dst[r * pitch..r * pitch + row_bytes]
                        .copy_from_slice(&pattern[r * row_bytes..(r + 1) * row_bytes]);
                }
            }
            let v = t.map_with(CpuAccess::Read).unwrap();
            let rows = t.layout().texture_height;
            let row_bytes = t.layout().tight_row_bytes();
            for r in 0..rows {
                assert_eq!(
                    &v.as_slice()[r * pitch..r * pitch + row_bytes],
                    &pattern[r * row_bytes..(r + 1) * row_bytes],
                    "{fmt:?}/{dt:?} row {r}"
                );
            }
        }
    }

    /// The sizes the semi-planar stride rule is proved at: three formats at
    /// one even size, plus an odd size whose chroma pairs need the even
    /// rounding on top of whatever the driver pads to.
    fn nv_sizes() -> [(PixelFormat, usize, usize); 4] {
        [
            (PixelFormat::Nv12, 64, 64),
            (PixelFormat::Nv16, 64, 64),
            (PixelFormat::Nv24, 64, 64),
            (PixelFormat::Nv12, 321, 241),
        ]
    }

    fn nv_tensor(
        fmt: PixelFormat,
        w: usize,
        h: usize,
        access: CpuAccess,
    ) -> D3d11TextureTensor<u8> {
        let shape = fmt.allocation_shape(w, h).unwrap();
        D3d11TextureTensor::<u8>::new_image(w, h, fmt, DType::U8, &shape, None, access)
            .unwrap_or_else(|e| panic!("{fmt:?} {w}x{h}: {e}"))
    }

    /// A semi-planar texture is exactly as wide as the row pitch a CPU map
    /// sees, so the combined plane's linear-byte model and the texture's texel
    /// grid are one number.
    ///
    /// The HAL writes luma and chroma lines at `row_stride` and the Path B
    /// shader wraps chroma addressing at the sampled texture's width. When the
    /// driver pads the staging pitch past the image width -- 128-byte aligned
    /// on this NVIDIA adapter -- a texture only `width` texels wide stores one
    /// producer row as two texture rows, and the shader samples past the row
    /// edge. An odd width breaks the same way on every adapter, because a
    /// chroma pair needs an even row.
    #[test]
    fn nv_textures_are_as_wide_as_their_pitch() {
        for (fmt, w, h) in nv_sizes() {
            let t = nv_tensor(fmt, w, h, CpuAccess::ReadWrite);
            let tex_w = t.layout().texture_width;
            let pitch = t
                .image_backing_row_stride()
                .expect("a readable tensor has staging");
            assert_eq!(tex_w, pitch, "{fmt:?} {w}x{h}: texture width vs map pitch");
            assert!(
                tex_w.is_multiple_of(2),
                "{fmt:?} {w}x{h}: texture width {tex_w} splits a chroma pair"
            );
            assert!(
                tex_w >= w,
                "{fmt:?} {w}x{h}: texture width {tex_w} is narrower than the image"
            );

            // The same size through the public constructor reports that width
            // as its row stride, or no stride at all when the driver needed no
            // padding and the texture is the natural row.
            let img = crate::Tensor::<u8>::image(
                w,
                h,
                fmt,
                Some(crate::TensorMemory::DmaBuf),
                CpuAccess::ReadWrite,
            )
            .unwrap_or_else(|e| panic!("{fmt:?} {w}x{h}: {e}"));
            assert_eq!(
                img.d3d11_layout().unwrap().texture_width,
                tex_w,
                "{fmt:?} {w}x{h}: Tensor::image chose another texture width"
            );
            let natural = w.next_multiple_of(2);
            let expected = (tex_w > natural).then_some(tex_w);
            assert_eq!(
                img.row_stride(),
                expected,
                "{fmt:?} {w}x{h}: recorded row stride (texture width {tex_w}, natural {natural})"
            );
        }
    }

    /// A semi-planar tensor with no staging still carries the stride rule: the
    /// allocator probes for a pitch whatever the `CpuAccess`, and the tensor
    /// reports that width as its backing pitch even with no staging texture to
    /// read one from. A texture nobody maps still has to carry the geometry
    /// every consumer of the combined plane assumes, and a write-only one has
    /// to be *told* the pitch or it lays its chroma lines out too tightly.
    #[test]
    fn nv_textures_with_cpu_access_none_share_the_stride_rule() {
        for (fmt, w, h) in nv_sizes() {
            let readable = nv_tensor(fmt, w, h, CpuAccess::ReadWrite);
            let expected = readable.layout().texture_width;
            for access in [CpuAccess::None, CpuAccess::Write] {
                let other = nv_tensor(fmt, w, h, access);
                assert_eq!(
                    other.layout().texture_width,
                    expected,
                    "{fmt:?} {w}x{h} with {access:?}"
                );
                assert_eq!(
                    other.image_backing_row_stride(),
                    Some(expected),
                    "{fmt:?} {w}x{h} with {access:?} reports no backing pitch"
                );
            }
        }
    }

    /// A write-only NV tensor's chroma lines land where the texture keeps
    /// them.
    ///
    /// The end of the rule above: a producer writes at the stride
    /// `Tensor::image` recorded, the host buffer is uploaded at the texture's
    /// own pitch, and a reader of the same texture finds the lines at the same
    /// offsets. NV24 is the shape that separates a right pitch from a wrong
    /// one, because its chroma line is `2W` bytes -- twice the image row -- and
    /// the model places line `i` two rows below line `i - 1`.
    #[test]
    fn a_write_only_nv_tensor_uploads_its_chroma_lines_at_the_texture_pitch() {
        let (w, h) = (64usize, 64usize);
        let fmt = PixelFormat::Nv24;
        let t = crate::Tensor::<u8>::image(
            w,
            h,
            fmt,
            Some(crate::TensorMemory::DmaBuf),
            CpuAccess::Write,
        )
        .unwrap();
        let stride = t
            .effective_row_stride()
            .expect("an image tensor knows its stride");
        let texture = t.d3d11_texture().expect("a texture-backed tensor");
        assert_eq!(
            t.row_stride(),
            (stride > w).then_some(stride),
            "a padded texture records its width as the row stride"
        );

        // Chroma line `i` is `2 * w` bytes at combined-plane row `h + 2 * i`,
        // which is where `ChromaLayout::uv_rows_per_luma` puts it and where the
        // Path B shader reads it.
        let line = 2 * w;
        let chroma_at = |i: usize| (h + 2 * i) * stride;
        let byte = |i: usize, k: usize| ((i * 5 + k * 3) % 251) as u8;
        {
            let mut m = t.map_write().unwrap();
            let dst = m.as_mut_slice();
            for i in 0..h {
                let at = chroma_at(i);
                for k in 0..line {
                    dst[at + k] = byte(i, k);
                }
            }
        }

        // A second, readable wrapper of the same texture: the only way to see
        // what the upload actually put there.
        let shape = fmt.allocation_shape(w, h).unwrap();
        // SAFETY: `t` holds the texture alive across this call.
        let r = unsafe {
            D3d11TextureTensor::<u8>::from_d3d11_texture(
                texture,
                w,
                h,
                fmt,
                DType::U8,
                &shape,
                CpuAccess::Read,
                None,
            )
        }
        .unwrap();
        assert_eq!(
            r.image_backing_row_stride(),
            Some(stride),
            "the reader must address the same rows as the writer"
        );
        let m = r.map_with(CpuAccess::Read).unwrap();
        let src = m.as_slice();
        for i in 0..h {
            let at = chroma_at(i);
            let expected: Vec<u8> = (0..line).map(|k| byte(i, k)).collect();
            assert_eq!(
                &src[at..at + line],
                expected.as_slice(),
                "chroma line {i} did not land at row {} of a {stride}-wide texture",
                h + 2 * i
            );
        }
    }

    /// An external semi-planar texture is accepted when it is at least as wide
    /// as an even image row, and its own width becomes the tensor's stride:
    /// the host allocated the padding, so the host's width is the pitch.
    #[test]
    fn an_external_semi_planar_texture_is_wrapped_at_its_own_width() {
        // 128 is a multiple of every staging alignment either adapter uses, so
        // this texture is 128 texels wide on both.
        let backing = nv_tensor(PixelFormat::Nv12, 128, 64, CpuAccess::None);
        assert_eq!(backing.layout().texture_width, 128);
        let shape = PixelFormat::Nv12.allocation_shape(100, 64).unwrap();
        // SAFETY: `backing` holds the texture live across this call.
        let t = unsafe {
            D3d11TextureTensor::<u8>::from_d3d11_texture(
                backing.texture_ptr(),
                100,
                64,
                PixelFormat::Nv12,
                DType::U8,
                &shape,
                CpuAccess::Read,
                None,
            )
        }
        .unwrap();
        assert_eq!(
            t.layout().texture_width,
            128,
            "the host's texture width is the wrapped tensor's stride"
        );
    }

    /// An external semi-planar texture that is not already as wide as its own
    /// staging pitch is refused: maps and pins of the wrapped tensor are laid
    /// out at that pitch, so accepting one would report a width the tensor
    /// does not address by.
    ///
    /// A raw 100-wide R8 texture is the shape a third party would hand over
    /// for a 100x64 NV12 image. Whether it is legal depends on the adapter,
    /// not on the HAL: this NVIDIA adapter gives it a 128-byte staging pitch
    /// and it is refused, while WARP's pitch is tight and it is accepted. The
    /// test measures the pitch and asserts the matching outcome rather than
    /// skipping on one of them.
    #[test]
    fn an_external_semi_planar_texture_narrower_than_its_pitch_is_refused() {
        let (w, h) = (100usize, 64usize);
        let fmt = PixelFormat::Nv12;
        let d = device().unwrap();
        // What a producer outside the HAL would create: the combined plane at
        // exactly the image's own width.
        let raw = D3d11ImageLayout {
            texture_width: w,
            ..image_d3d11_layout(fmt, DType::U8, w, h).unwrap()
        };
        assert_eq!(raw.texture_height, 96, "NV12 64 rows plus 32 chroma rows");
        let tex =
            create_texture(d, &texture_desc(&raw, D3D11_USAGE_DEFAULT, BIND, 0, MISC)).unwrap();
        let pitch = staging_row_pitch(d, &create_staging_texture(d, &raw).unwrap()).unwrap();

        // SAFETY: `tex` is live for the whole call.
        let wrapped = unsafe {
            crate::Tensor::<u8>::from_d3d11_texture(tex.as_raw(), w, h, fmt, CpuAccess::Read, None)
        };
        if pitch == w {
            let t = wrapped.expect("a texture already at its own pitch is wrappable");
            assert_eq!(t.effective_row_stride(), Some(w));
        } else {
            let err = wrapped.unwrap_err();
            assert!(
                matches!(err, Error::InvalidArgument(_))
                    && err.to_string().contains("driver row pitch"),
                "{err}"
            );
        }
    }

    /// A wrapped semi-planar tensor reports the texture's width as its row
    /// stride, exactly as an HAL-allocated one does.
    ///
    /// Without it the tensor would report the natural `even(width)` stride
    /// while its texture is the driver's pitch wide, and the Path B shader --
    /// which takes its `tex_width` from the stride -- would sample a padded
    /// texture at the image width. Both constructors record it, so a texture
    /// that crosses a process boundary as a shared handle lands the same way
    /// as one passed by pointer.
    #[test]
    fn wrapped_nv_tensors_record_the_textures_width_as_their_row_stride() {
        for (w, h) in [(321usize, 241usize), (64, 64)] {
            let fmt = PixelFormat::Nv12;
            let src = nv_tensor(fmt, w, h, CpuAccess::None);
            let texture_width = src.layout().texture_width;

            // SAFETY: `src` holds the texture live across the call.
            let by_ptr = unsafe {
                crate::Tensor::<u8>::from_d3d11_texture(
                    src.texture_ptr(),
                    w,
                    h,
                    fmt,
                    CpuAccess::Read,
                    None,
                )
            }
            .unwrap_or_else(|e| panic!("{fmt:?} {w}x{h} by pointer: {e}"));
            assert_eq!(
                by_ptr.effective_row_stride(),
                Some(texture_width),
                "{fmt:?} {w}x{h} wrapped by pointer"
            );

            let handle = src.shared_handle().unwrap();
            // SAFETY: `handle` is a shared NT handle this test owns.
            let by_handle = unsafe {
                crate::Tensor::<u8>::from_d3d11_shared_handle(
                    handle.as_raw_handle(),
                    w,
                    h,
                    fmt,
                    CpuAccess::Read,
                    None,
                    None,
                )
            }
            .unwrap_or_else(|e| panic!("{fmt:?} {w}x{h} by handle: {e}"));
            assert_eq!(
                by_handle.effective_row_stride(),
                Some(texture_width),
                "{fmt:?} {w}x{h} wrapped by shared handle"
            );
        }
    }

    /// A texture narrower than an even image row cannot hold the image, so it
    /// is refused rather than wrapped at a width whose last chroma pair falls
    /// off the row.
    #[test]
    fn an_external_semi_planar_texture_narrower_than_the_image_is_refused() {
        let backing = nv_tensor(PixelFormat::Nv12, 128, 64, CpuAccess::None);
        let shape = PixelFormat::Nv12.allocation_shape(129, 64).unwrap();
        // SAFETY: `backing` holds the texture live across this call.
        let err = unsafe {
            D3d11TextureTensor::<u8>::from_d3d11_texture(
                backing.texture_ptr(),
                129,
                64,
                PixelFormat::Nv12,
                DType::U8,
                &shape,
                CpuAccess::Read,
                None,
            )
        }
        .unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)), "{err}");
    }

    #[test]
    fn write_only_uses_the_shadow_and_lands_in_the_texture() {
        let shape = PixelFormat::Rgba.allocation_shape(64, 32).unwrap();
        let w = rgba(64, 32, CpuAccess::Write);
        assert!(
            w.image_backing_row_stride().is_none(),
            "no staging, tight rows"
        );
        {
            let mut v = w.map_with(CpuAccess::Write).unwrap();
            v.as_mut_slice().fill(0xA5);
        }
        // Read it back through a second, readable wrapper of the same texture.
        // SAFETY: `w` holds the texture alive across this call.
        let r = unsafe {
            D3d11TextureTensor::<u8>::from_d3d11_texture(
                w.texture_ptr(),
                64,
                32,
                PixelFormat::Rgba,
                DType::U8,
                &shape,
                CpuAccess::Read,
                None,
            )
        }
        .unwrap();
        let v = r.map_with(CpuAccess::Read).unwrap();
        assert!(v.as_slice()[..64 * 4].iter().all(|&b| b == 0xA5));
    }

    #[test]
    fn hardware_only_tensors_refuse_maps_and_have_no_staging() {
        let t = rgba(64, 32, CpuAccess::None);
        assert!(matches!(
            t.map_with(CpuAccess::Read).unwrap_err(),
            Error::InvalidOperation(_)
        ));
        assert!(t.image_backing_row_stride().is_none());
    }

    #[test]
    fn shared_handle_opens_on_a_second_device_and_reads_the_bytes() {
        let t = rgba(64, 32, CpuAccess::ReadWrite);
        {
            let mut v = t.map_with(CpuAccess::Write).unwrap();
            v.as_mut_slice().fill(0x5A);
        }
        // The second device is not ordered against this one by anything but
        // the shared fence, so wait for the write to land before reading.
        let value = device().unwrap().signal().expect("fence signal");
        let event = device().unwrap().event_for(value).unwrap();
        // SAFETY: `event` is the live event handle `event_for` just returned.
        let waited = unsafe { WaitForSingleObject(HANDLE(event.as_raw_handle()), 5_000) };
        assert_eq!(waited, WAIT_OBJECT_0, "fence value {value} within 5 s");

        let h = t.shared_handle().unwrap();
        let other = super::tests_support::second_device();
        let bytes =
            super::tests_support::read_through(&other, HANDLE(h.as_raw_handle()), 64, 32, 4);
        assert!(bytes.iter().all(|&b| b == 0x5A));
    }

    #[test]
    fn completion_is_absent_until_recorded_then_carries_the_value() {
        let t = rgba(8, 8, CpuAccess::None);
        assert!(t.gpu_completion().unwrap().is_none());
        let v = device().unwrap().signal().unwrap();
        t.set_gpu_write(v);
        let c = t.gpu_completion().unwrap().unwrap();
        assert_eq!(c.value, v);
    }

    #[test]
    fn a_map_does_not_block_on_a_recorded_write_the_fence_has_passed() {
        // The wait a map queues is GPU-side, and a value already complete
        // retires at once, so recording an old value must not cost the
        // caller anything. Without the bound this would be the failure mode
        // of getting the ordering wrong: a wait on a value nothing will ever
        // signal, paid for on every map.
        let t = rgba(16, 16, CpuAccess::ReadWrite);
        let v = device().unwrap().signal().unwrap();
        let ev = device().unwrap().event_for(v).unwrap();
        // SAFETY: `ev` is a live event handle owned by this scope.
        let w = unsafe { WaitForSingleObject(HANDLE(ev.as_raw_handle()), 5000) };
        assert_eq!(w, WAIT_OBJECT_0, "the signalled value completes");
        assert!(device().unwrap().completed_value() >= v);
        t.set_gpu_write(v);
        let started = Instant::now();
        let mapped = t.map_with(CpuAccess::Read).unwrap();
        let elapsed = started.elapsed();
        drop(mapped);
        assert!(
            elapsed < Duration::from_secs(2),
            "mapping over an already-complete recorded write took {elapsed:?}"
        );
    }

    #[test]
    fn a_map_sees_a_second_device_write_the_recorded_completion_covers() {
        // A producer this device does not order itself against: a second
        // D3D11 device writes the texture through its shared handle and
        // signals a fence of its own behind that write. The import opens
        // that fence, queues a wait on it, and re-signals the local
        // timeline behind the wait, so the value this tensor records is
        // reached only once the second device's bytes are there -- and the
        // map has to be behind it.
        //
        // The earlier shape of this test queued the write on the same
        // immediate context the staging refresh uses, where context order
        // alone already guaranteed the result; this one carries the
        // ordering on a fence and nothing else.
        assert!(
            device().unwrap().fence_is_shared(),
            "this host's copy signals the process's shared fence, so the map \
             path's wait is live rather than skipped"
        );
        // Big enough, and preceded by enough zero fills, that the write is
        // still in flight when the map below asks for the bytes: 16 MiB of
        // uploads in front of the 1 MiB that matters.
        const W: usize = 512;
        const H: usize = 512;
        let shape = PixelFormat::Rgba.allocation_shape(W, H).unwrap();
        let t = rgba(W, H, CpuAccess::ReadWrite);
        let h = t.shared_handle().unwrap();
        let other = super::tests_support::second_device();
        let (fence, fence_handle) = super::tests_support::shared_fence(&other);
        super::tests_support::write_through(
            &other,
            HANDLE(h.as_raw_handle()),
            W,
            H,
            4,
            0x5A,
            16,
            &fence,
            1,
        );
        // SAFETY: both handles are owned by this scope and valid in this
        // process for the duration of the call.
        let imported = unsafe {
            D3d11TextureTensor::<u8>::from_d3d11_shared_handle(
                h.as_raw_handle(),
                W,
                H,
                PixelFormat::Rgba,
                DType::U8,
                &shape,
                CpuAccess::Read,
                Some((fence_handle.as_raw_handle(), 1)),
                Some("second-device producer"),
            )
        }
        .unwrap();
        let mapped = imported.map_with(CpuAccess::Read).unwrap();
        let pitch = imported.image_backing_row_stride().unwrap();
        let row_bytes = imported.layout().tight_row_bytes();
        for r in 0..imported.layout().texture_height {
            assert!(
                mapped.as_slice()[r * pitch..r * pitch + row_bytes]
                    .iter()
                    .all(|&b| b == 0x5A),
                "row {r} of the second device's write"
            );
        }
    }

    #[test]
    fn a_recorded_write_above_the_counter_does_not_block_the_next_map() {
        // `ef_tensor_set_gpu_write` takes any u64, so a caller can record a
        // value this device's fence will never reach. Queuing a wait on it
        // would not stall one map: it stalls the immediate context for the
        // rest of the process, and every later signal and every other
        // tensor's work with it. The map path has to ignore it instead.
        let t = rgba(32, 32, CpuAccess::ReadWrite);
        {
            let mut v = t.map_with(CpuAccess::Write).unwrap();
            v.as_mut_slice().fill(0x77);
        }
        let unreachable = device().unwrap().last_signalled() + 1_000_000;
        t.set_gpu_write(unreachable);

        // try_map rather than map: were the bound gone, a blocking map would
        // hang this process instead of failing, and a hung test says less
        // than a failed one. The loop is the same shape
        // `try_map_eventually_succeeds_and_returns_the_written_bytes` uses --
        // bounded by the clock, yielding between attempts.
        let deadline = Instant::now() + Duration::from_secs(5);
        let mapped = loop {
            match t.try_map_with(CpuAccess::Read) {
                Ok(v) => break v,
                Err(Error::IoError(e)) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(e) => panic!("try_map_with returned an unexpected error: {e}"),
            }
            assert!(
                Instant::now() < deadline,
                "a recorded write of {unreachable}, above the counter's {}, blocked \
                 the map, so a wait was queued on a value nothing will ever signal",
                device().unwrap().last_signalled()
            );
            std::thread::yield_now();
        };
        let row_bytes = t.layout().tight_row_bytes();
        assert!(
            mapped.as_slice()[..row_bytes].iter().all(|&b| b == 0x77),
            "the map still reads the texture, it just does not wait on the value"
        );
        drop(mapped);
        // Ignored by the map path, not discarded: `gpu_completion` describes
        // the producer's timeline, and a consumer opening the producer's
        // fence is entitled to the value the producer named.
        assert_eq!(t.last_gpu_write(), unreachable);
    }

    #[test]
    fn from_shared_handle_round_trips_and_honours_a_completion() {
        let shape = PixelFormat::Rgba.allocation_shape(16, 16).unwrap();
        let src = rgba(16, 16, CpuAccess::ReadWrite);
        {
            let mut v = src.map_with(CpuAccess::Write).unwrap();
            v.as_mut_slice().fill(7);
        }
        let v = device().unwrap().signal().unwrap();
        src.set_gpu_write(v);
        let h = src.shared_handle().unwrap();
        let c = src.gpu_completion().unwrap().unwrap();
        // SAFETY: `h` and `c.fence` are handles this process owns, live for
        // the call.
        let dst = unsafe {
            D3d11TextureTensor::<u8>::from_d3d11_shared_handle(
                h.as_raw_handle(),
                16,
                16,
                PixelFormat::Rgba,
                DType::U8,
                &shape,
                CpuAccess::Read,
                Some((c.fence.as_raw_handle(), c.value)),
                None,
            )
        }
        .unwrap();
        // Not the producer's value: the import queues a wait on the producer's
        // fence and then signals the *local* one behind it, so what it records
        // is a value on the timeline `gpu_completion` and the CUDA semaphore
        // wait both read. In-process the two fences are the same object, so the
        // recorded value is simply a newer point on it.
        assert!(
            dst.last_gpu_write() > v,
            "the import recorded {} on the local timeline, past the producer's {v}",
            dst.last_gpu_write()
        );
        let ev = device().unwrap().event_for(dst.last_gpu_write()).unwrap();
        // SAFETY: `ev` is a live event handle owned by this scope.
        let w = unsafe { WaitForSingleObject(HANDLE(ev.as_raw_handle()), 5000) };
        assert_eq!(w, WAIT_OBJECT_0, "the recorded value is reachable");
        let mapped = dst.map_with(CpuAccess::Read).unwrap();
        let pitch = dst.image_backing_row_stride().unwrap();
        let row_bytes = dst.layout().tight_row_bytes();
        for r in 0..dst.layout().texture_height {
            assert!(
                mapped.as_slice()[r * pitch..r * pitch + row_bytes]
                    .iter()
                    .all(|&b| b == 7),
                "row {r} of the shared-handle round trip"
            );
        }
    }

    #[test]
    fn external_texture_on_a_mismatched_description_is_refused() {
        let shape = PixelFormat::Rgba.allocation_shape(16, 16).unwrap();
        let t = rgba(16, 16, CpuAccess::None);
        // SAFETY: `t` holds the texture alive across this call.
        let err = unsafe {
            D3d11TextureTensor::<u8>::from_d3d11_texture(
                t.texture_ptr(),
                32,
                16,
                PixelFormat::Rgba,
                DType::U8,
                &shape,
                CpuAccess::None,
                None,
            )
        }
        .unwrap_err();
        assert!(matches!(err, crate::Error::InvalidArgument(_)));
    }

    #[test]
    fn second_reader_shares_the_map_and_a_writer_is_refused_while_mapped() {
        let t = rgba(16, 16, CpuAccess::ReadWrite);
        {
            let mut v = t.map_with(CpuAccess::Write).unwrap();
            v.as_mut_slice().fill(0x3C);
        }
        let row = t.layout().tight_row_bytes();
        let a = t.map_with(CpuAccess::Read).unwrap();
        let b = t.map_with(CpuAccess::Read).unwrap();
        // One mapping, two readers: the same address and the same bytes.
        assert_eq!(a.as_slice().as_ptr(), b.as_slice().as_ptr());
        assert_eq!(a.as_slice()[..row], b.as_slice()[..row]);
        assert!(a.as_slice()[..row].iter().all(|&x| x == 0x3C));

        assert!(matches!(
            t.map_with(CpuAccess::Write).unwrap_err(),
            Error::InvalidOperation(_)
        ));
        drop(a);
        assert!(
            matches!(
                t.map_with(CpuAccess::Write).unwrap_err(),
                Error::InvalidOperation(_)
            ),
            "the second reader still holds the mapping"
        );
        drop(b);

        let mut w = t.map_with(CpuAccess::Write).unwrap();
        w.as_mut_slice()[..row].fill(0x4D);
        drop(w);
        assert!(t.map_with(CpuAccess::Read).unwrap().as_slice()[..row]
            .iter()
            .all(|&x| x == 0x4D));
    }

    #[test]
    fn a_view_shares_the_map_state_with_its_parent() {
        let shape = PixelFormat::Rgba.allocation_shape(16, 16).unwrap();
        let t = rgba(16, 16, CpuAccess::ReadWrite);
        let sub = t.view(0, &shape).unwrap();
        let row = t.layout().tight_row_bytes();

        let a = t.map_with(CpuAccess::Read).unwrap();
        let b = sub.map_with(CpuAccess::Read).unwrap();
        assert_eq!(a.as_slice().as_ptr(), b.as_slice().as_ptr());
        assert!(matches!(
            sub.map_with(CpuAccess::Write).unwrap_err(),
            Error::InvalidOperation(_)
        ));
        assert!(matches!(
            t.map_with(CpuAccess::Write).unwrap_err(),
            Error::InvalidOperation(_)
        ));
        drop(a);
        drop(b);

        let mut w = sub.map_with(CpuAccess::Write).unwrap();
        w.as_mut_slice()[..row].fill(0x71);
        drop(w);
        assert!(t.map_with(CpuAccess::Read).unwrap().as_slice()[..row]
            .iter()
            .all(|&x| x == 0x71));
    }

    #[test]
    fn host_pin_keeps_one_address_across_syncs() {
        let t = rgba(16, 16, CpuAccess::ReadWrite);
        let row = t.layout().tight_row_bytes();
        let pin = t.host_pin(CpuAccess::ReadWrite).unwrap();
        let address = pin.as_ptr();

        {
            let mut v = t.map_with(CpuAccess::Write).unwrap();
            v.as_mut_slice().fill(0x5E);
        }
        t.sync_for_cpu(CpuAccess::Read).unwrap();
        assert_eq!(pin.as_ptr(), address, "a pin's address survives a sync");
        // SAFETY: no device write is in flight and `sync_for_cpu` has just
        // refreshed the buffer.
        assert!(unsafe { pin.as_slice() }[..row].iter().all(|&b| b == 0x5E));

        // SAFETY: as above; this test is the only holder of the buffer.
        unsafe { std::ptr::write_bytes(pin.as_mut_ptr(), 0x27, pin.len()) };
        t.sync_for_device(CpuAccess::Write).unwrap();
        assert_eq!(pin.as_ptr(), address, "and survives an upload");

        let v = t.map_with(CpuAccess::Read).unwrap();
        assert!(v.as_slice()[..row].iter().all(|&b| b == 0x27));
    }

    /// A map and a pin address the same rows at the same pitch.
    ///
    /// Every byte a map or a pin hands out is in backing (pitched) space, and
    /// `image_backing_row_stride()` -- which `Tensor::image` records as the
    /// tensor's `row_stride` -- is that pitch. A consumer writing row `y` at
    /// `y * row_stride` through a map therefore reads it back at the same
    /// offset through a pin, with no repacking anywhere between. That is the
    /// rule the image engine's GL readback depends on: it packs its rows
    /// tightly and re-spaces them to this pitch, and a backing that quietly
    /// packed one of the two windows differently would put the two halves of
    /// that contract out of step.
    ///
    /// An RGB image is the shape that separates them: its tight row is
    /// `width * 3`, which no 128-byte-aligned driver pitch is a multiple of, so
    /// on a discrete adapter the two numbers differ. On the software adapter
    /// the pitch is tight and the test still holds, trivially.
    #[test]
    fn a_map_and_a_pin_agree_row_for_row_at_the_backing_pitch() {
        let w = 64;
        let h = 48;
        let shape = PixelFormat::Rgb.allocation_shape(w, h).unwrap();
        let t = D3d11TextureTensor::<u8>::new_image(
            w,
            h,
            PixelFormat::Rgb,
            DType::U8,
            &shape,
            None,
            CpuAccess::ReadWrite,
        )
        .unwrap();
        let pitch = t
            .image_backing_row_stride()
            .expect("a readable tensor has staging");
        let row_bytes = t.layout().tight_row_bytes();
        assert!(
            pitch >= row_bytes,
            "the staging pitch must cover a whole row"
        );

        // One distinct byte per row, written at the backing pitch.
        {
            let mut v = t.map_with(CpuAccess::Write).unwrap();
            let buf = v.as_mut_slice();
            assert_eq!(buf.len(), pitch * h, "a map covers the pitched rows");
            for y in 0..h {
                buf[y * pitch..y * pitch + row_bytes].fill(y as u8);
            }
        }

        let pin = t.host_pin(CpuAccess::ReadWrite).unwrap();
        t.sync_for_cpu(CpuAccess::Read).unwrap();
        // SAFETY: no device write is in flight and `sync_for_cpu` has just
        // refreshed the buffer; this test is the only holder.
        let bytes = unsafe { pin.as_slice() };
        for y in 0..h {
            assert!(
                bytes[y * pitch..y * pitch + row_bytes]
                    .iter()
                    .all(|&b| b == y as u8),
                "row {y} of the pin is not the row the map wrote at pitch {pitch}"
            );
        }
    }

    /// `crate::pin::HostPin::alignment` documents 64 bytes as what an
    /// inference runtime's custom allocation needs; the allocator alone does
    /// not provide it, so `HostBuffer` carries the slack to reach it.
    #[test]
    fn a_host_pin_is_64_byte_aligned() {
        // A pitch that is not itself a multiple of 64 would still land on an
        // aligned base, because the base is aligned, not the pitch.
        for (w, h) in [(64usize, 64usize), (37, 5), (1, 1)] {
            let t = rgba(w, h, CpuAccess::ReadWrite);
            let pin = t.host_pin(CpuAccess::ReadWrite).unwrap();
            assert!(
                pin.alignment() >= 64,
                "{w}x{h}: pin is {}-byte aligned",
                pin.alignment()
            );
        }
    }

    /// A map covers this tensor's own rows at the backing pitch -- for a
    /// sub-view, its own window rather than everything left in the parent.
    /// A batched destination's per-band map depends on this to match the same
    /// convert done standalone.
    #[test]
    fn a_map_covers_this_tensors_rows_not_the_parents_tail() {
        let t = rgba(32, 8, CpuAccess::ReadWrite);
        let pitch = t.image_backing_row_stride().unwrap();

        let whole = t.map_with(CpuAccess::Read).unwrap();
        assert_eq!(whole.as_slice().len(), pitch * 8);
        drop(whole);

        // A window two rows in, two rows tall: its map covers those two rows,
        // not the six that remain in the parent.
        let v = t.view(pitch * 2, &[2, 32, 4]).unwrap();
        let mapped = v.map_with(CpuAccess::Read).unwrap();
        assert_eq!(mapped.as_slice().len(), pitch * 2);
        drop(mapped);

        // An explicit request past the window is refused rather than clamped.
        assert!(matches!(
            v.map_with_byte_size(pitch * 7, CpuAccess::Read),
            Err(Error::InsufficientCapacity { .. })
        ));
    }

    #[test]
    fn try_map_eventually_succeeds_and_returns_the_written_bytes() {
        let t = rgba(64, 64, CpuAccess::ReadWrite);
        let row = t.layout().tight_row_bytes();
        {
            let mut v = t.map_with(CpuAccess::Write).unwrap();
            v.as_mut_slice().fill(0x6B);
        }

        // The first attempt either maps or reports that the copy is still in
        // flight. Any other error is a failure, not a retry.
        match t.try_map_with(CpuAccess::Read) {
            Ok(_) => {}
            Err(Error::IoError(e)) if e.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(e) => panic!("first try_map_with returned an unexpected error: {e}"),
        }

        // Bounded by the clock alone, and yielding between attempts. An
        // attempt cap is the wrong bound here: 10 000 spins elapse in tens of
        // milliseconds, so the loop could exhaust it before the GPU had a
        // chance to finish the copy the first attempt submitted -- and a tight
        // spin makes that worse, not better, because the threads that finish
        // the copy (the driver's, and on WARP the CPU workers that *are* the
        // GPU) are competing with this loop for the same core.
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut attempts = 0usize;
        let mapped = loop {
            attempts += 1;
            match t.try_map_with(CpuAccess::Read) {
                Ok(v) => break v,
                Err(Error::IoError(e)) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(e) => panic!("try_map_with returned an unexpected error: {e}"),
            }
            assert!(
                Instant::now() < deadline,
                "try_map_with never succeeded within 2s ({attempts} attempts)"
            );
            std::thread::yield_now();
        };
        assert!(mapped.as_slice()[..row].iter().all(|&b| b == 0x6B));
    }

    #[test]
    fn read_map_of_a_write_only_tensor_does_not_upload() {
        let shape = PixelFormat::Rgba.allocation_shape(64, 32).unwrap();
        let w = rgba(64, 32, CpuAccess::Write);
        {
            let mut v = w.map_with(CpuAccess::Write).unwrap();
            v.as_mut_slice().fill(0xA5);
        }
        // A readable wrapper of the same texture, to observe and to overwrite.
        // SAFETY: `w` holds the texture alive across this call.
        let r = unsafe {
            D3d11TextureTensor::<u8>::from_d3d11_texture(
                w.texture_ptr(),
                64,
                32,
                PixelFormat::Rgba,
                DType::U8,
                &shape,
                CpuAccess::ReadWrite,
                None,
            )
        }
        .unwrap();
        let pitch = r.image_backing_row_stride().unwrap();
        let row = r.layout().tight_row_bytes();
        assert!(r.map_with(CpuAccess::Read).unwrap().as_slice()[..row]
            .iter()
            .all(|&b| b == 0xA5));

        // A read map of the write-only tensor must publish nothing.
        drop(w.map_with(CpuAccess::Read).unwrap());

        {
            let mut v = r.map_with(CpuAccess::Write).unwrap();
            let dst = v.as_mut_slice();
            for n in 0..32 {
                dst[n * pitch..n * pitch + row].fill(0x11);
            }
        }
        // Taken again after the overwrite: this is the ordering that catches
        // an unconditional upload, which would put 0xA5 back.
        drop(w.map_with(CpuAccess::Read).unwrap());

        assert!(
            r.map_with(CpuAccess::Read).unwrap().as_slice()[..row]
                .iter()
                .all(|&b| b == 0x11),
            "a read map of a write-only tensor overwrote the texture"
        );
    }

    #[test]
    fn view_at_a_row_offset_maps_the_right_bytes() {
        let t = rgba(16, 16, CpuAccess::ReadWrite);
        let pitch = t.image_backing_row_stride().unwrap();
        let row = t.layout().tight_row_bytes();
        {
            let mut v = t.map_with(CpuAccess::Write).unwrap();
            let dst = v.as_mut_slice();
            for n in 0..16 {
                dst[n * pitch..n * pitch + row].fill(0xB0 + n as u8);
            }
        }
        // Offsets are in backing space, so three whole rows is `3 * pitch`.
        let sub = t.view(3 * pitch, &[13, 16, 4]).unwrap();
        assert_eq!(sub.capacity_bytes(), 16 * pitch - 3 * pitch);
        let v = sub.map_with(CpuAccess::Read).unwrap();
        for n in 0..13 {
            assert!(
                v.as_slice()[n * pitch..n * pitch + row]
                    .iter()
                    .all(|&b| b == 0xB0 + (n + 3) as u8),
                "view row {n} should be texture row {}",
                n + 3
            );
        }
    }
}
