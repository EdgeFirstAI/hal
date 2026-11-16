// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//! `ID3D11Texture2D`-backed storage. GPU consumers read the texture directly;
//! the CPU reads and writes through a staging copy, or, for write-only
//! tensors, through a host buffer uploaded with `UpdateSubresource`.
//!
//! [`D3d11TextureTensor::host_pin`] hands out an address into the host buffer,
//! for every access kind. `sync_for_cpu` and `sync_for_device` move bytes
//! between that buffer and the texture.

use super::com::hr;
use super::device::{
    device, duplicate_handle, duplicate_raw_handle, wait_cpu_for, D3d11Device, GpuCompletion,
};
use crate::d3d11_layout::{
    combined_semi_planar_layout, image_d3d11_layout, D3d11ImageLayout, DXGI_FORMAT_R8_UNORM,
};
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
    ID3D11Fence, ID3D11Texture2D, D3D11_BIND_RENDER_TARGET, D3D11_BIND_SHADER_RESOURCE, D3D11_BOX,
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

/// Bind flags every allocated texture carries: render target and shader
/// resource.
const BIND: u32 = (D3D11_BIND_RENDER_TARGET.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32;
/// Misc flags every allocated texture carries: shared, with an NT handle.
const MISC: u32 = (D3D11_RESOURCE_MISC_SHARED.0 | D3D11_RESOURCE_MISC_SHARED_NTHANDLE.0) as u32;

/// Who holds a CPU window onto a backing. Readers share one mapping; a writer
/// is exclusive.
#[derive(Default)]
struct MapState {
    /// Live read maps sharing one mapping.
    readers: usize,
    /// A writable map is live. Never true while `readers > 0`.
    writer: bool,
    /// Base address of the live mapping, meaningful only while `readers > 0`
    /// or `writer`.
    base: usize,
    /// A `CopyResource` into the staging texture is queued and no `Map` has
    /// consumed it yet.
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

    /// Gives a claim back. Returns whether that was the last holder.
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
    /// The allocation, over-sized by `HOST_PIN_ALIGN - 1` bytes. Owns the
    /// bytes [`Self::base`] points at.
    _bytes: UnsafeCell<Box<[u8]>>,
    /// Aligned address of the first row, taken once in [`Self::new`].
    base: *mut u8,
    /// Bytes of rows, excluding the alignment slack.
    len: usize,
    row_pitch: usize,
    state: Mutex<MapState>,
}

/// Alignment of the address [`D3d11TextureTensor::host_pin`] hands out.
const HOST_PIN_ALIGN: usize = 64;

// SAFETY: `MapState` gives the buffer one writer at a time, and an upload
// reads it under the same claim. A `host_pin` holder carries the coherency
// obligation `crate::pin` documents.
unsafe impl Sync for HostBuffer {}
// SAFETY: a `Box<[u8]>` has no thread affinity.
unsafe impl Send for HostBuffer {}

impl HostBuffer {
    fn new(row_pitch: usize, rows: usize) -> Self {
        let len = row_pitch * rows;
        let mut bytes = vec![0u8; len + HOST_PIN_ALIGN - 1].into_boxed_slice();
        let start = bytes.as_mut_ptr();
        // SAFETY: the offset is at most `HOST_PIN_ALIGN - 1`, the slack
        // allocated.
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

/// Where a CPU view of a native NV12 texture holding a shorter image keeps the
/// image's chroma rows.
///
/// The texture's chroma plane starts at row `texture_row` of a mapped copy
/// (its own height), while the image's combined plane has them right after its
/// own luma rows, at `image_row`. A map moves them up after the texture is
/// copied in, and a write moves them back down before it is copied out. Rows
/// between the two are not preserved.
#[derive(Debug, Clone, Copy)]
struct ChromaRepack {
    texture_row: usize,
    image_row: usize,
    rows: usize,
    pitch: usize,
}

impl ChromaRepack {
    /// Moves the chroma rows from the texture's plane to the image's.
    ///
    /// # Safety
    ///
    /// `base` must address `pitch * (texture_row + rows)` writable bytes.
    unsafe fn move_to_image(&self, base: *mut u8) {
        // SAFETY: both ranges are inside that span; `copy` allows overlap.
        unsafe {
            std::ptr::copy(
                base.add(self.texture_row * self.pitch),
                base.add(self.image_row * self.pitch),
                self.rows * self.pitch,
            )
        };
    }

    /// Moves the chroma rows from the image's plane to the texture's.
    ///
    /// # Safety
    ///
    /// As [`move_to_image`](Self::move_to_image).
    unsafe fn move_to_texture(&self, base: *mut u8) {
        // SAFETY: as in `move_to_image`.
        unsafe {
            std::ptr::copy(
                base.add(self.image_row * self.pitch),
                base.add(self.texture_row * self.pitch),
                self.rows * self.pitch,
            )
        };
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
    /// shared with every view.
    host: Arc<OnceLock<HostBuffer>>,
    nt_handle: Arc<OwnedHandle>,
    identity: BufferIdentity,
    last_gpu_write: Arc<AtomicU64>,
    /// The producer's fence, opened at import and kept for the tensor's
    /// lifetime. `None` for an allocated or wrapped texture, and for an import
    /// that carried no completion.
    imported_fence: Option<ID3D11Fence>,
    /// Set when `host_pin` hands out a writable pin. `sync_for_device` then
    /// publishes from the host buffer.
    wrote_host: Arc<AtomicBool>,
    view_offset: usize,
    _marker: PhantomData<T>,
}

/// How long an import blocks on a producer's fence when this device cannot
/// queue a GPU-side wait, in milliseconds.
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

/// The driver's row pitch for a staging texture, read from one map of it.
fn staging_row_pitch(d: &D3d11Device, tex: &ID3D11Texture2D) -> Result<usize> {
    let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
    // SAFETY: `tex` is a live staging texture and `mapped` is a valid
    // out-parameter.
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
        rows: layout.mapped_rows(),
        state: Mutex::new(MapState::default()),
    })
}

/// A semi-planar layout widened to the row pitch the driver gives its staging
/// copy: the texture's texel width equals the combined plane's row stride.
///
/// Probes one staging texture at the layout's width and, when its pitch is
/// padded, a second at that pitch. Fails when the second is padded again.
fn widen_semi_planar_to_pitch(
    d: &D3d11Device,
    layout: D3d11ImageLayout,
) -> Result<D3d11ImageLayout> {
    // R8: a pitch in bytes is a pitch in texels.
    let pitch = staging_row_pitch(d, &create_staging_texture(d, &layout)?)?;
    if pitch <= layout.tight_row_bytes() {
        return Ok(layout);
    }
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
/// `BIND_RENDER_TARGET` is not required.
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
    // An R8 texture for an even NV12 image takes the combined-plane layout.
    let layout = if layout.is_native_nv12() && desc.Format.0 as u32 == DXGI_FORMAT_R8_UNORM {
        combined_semi_planar_layout(format, width, height).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "external D3D11 texture: no combined-plane layout for {format:?} {width}x{height}"
            ))
        })?
    } else {
        layout
    };
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
    // A combined-plane texture's width is the image's row stride: a width
    // above the image's even row becomes the tensor's backing pitch. Every
    // other texture's width is its image row.
    let layout = if format.layout() == PixelLayout::SemiPlanar && !layout.is_native_nv12() {
        let texture_width = desc.Width as usize;
        if texture_width < layout.texture_width {
            return Err(Error::InvalidArgument(format!(
                "external D3D11 texture: texture is {texture_width} texels wide but \
                 {format:?}/{dtype:?} {width}x{height} needs at least {} (the image's even row)",
                layout.texture_width
            )));
        }
        // The texture's width has to be even and equal to its staging row
        // pitch.
        let wrapped = layout.widened_to(texture_width);
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
        // A combined-plane texture is widened to the row pitch a CPU map sees.
        let layout = if format.layout() == PixelLayout::SemiPlanar && !layout.is_native_nv12() {
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
    #[allow(clippy::too_many_arguments)]
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
        // `ID3D11Texture2D`; `cloned` takes the reference this tensor keeps.
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
    /// given, waits for it on the immediate context.
    ///
    /// # Safety
    ///
    /// `handle` must be a shared NT handle valid in this process, and the
    /// completion's handle, when given, a shared fence handle. Both stay owned
    /// by the caller: this duplicates what it keeps.
    #[allow(clippy::too_many_arguments)]
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
        // Keyed on the opened texture, not on the caller's handle value.
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
    /// Opens the producer's fence, queues a GPU-side wait on its value, and
    /// records a local signal taken after that wait as `last_gpu_write`. A
    /// device that cannot queue a wait or signal blocks on the producer's
    /// fence for up to `IMPORT_WAIT_MS` milliseconds and records nothing.
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
        // SAFETY: both interfaces are live.
        hr("ID3D11DeviceContext4::Wait", unsafe {
            ctx4.Wait(&fence, value)
        })?;
        self.imported_fence = Some(fence);
        self.last_gpu_write
            .store(d.signal().unwrap_or(0), Ordering::Release);
        Ok(())
    }

    /// Shared tail of the three constructors: builds the CPU-side backing
    /// `access` calls for and assembles the tensor.
    #[allow(clippy::too_many_arguments)]
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
        // Only a readable tensor gets a staging texture.
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

    /// How CPU views at `pitch` move this tensor's chroma rows, when it is a
    /// native NV12 texture holding a shorter image; `None` otherwise.
    fn chroma_repack(&self, pitch: usize) -> Option<ChromaRepack> {
        if !self.layout.is_native_nv12() {
            return None;
        }
        let [combined, _] = self.shape[..] else {
            return None;
        };
        let image_row = PixelFormat::Nv12.image_height_from_combined(combined)?;
        (image_row < self.layout.texture_height).then_some(ChromaRepack {
            texture_row: self.layout.texture_height,
            image_row,
            rows: combined - image_row,
            pitch,
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

    /// Bytes of one image row: what this tensor's shape counts per row.
    ///
    /// The texture row for every format except a semi-planar one, whose shape
    /// counts `width` samples per row while its texture is as wide as the row
    /// stride.
    fn image_row_bytes(&self) -> usize {
        if self.format.layout() == PixelLayout::SemiPlanar {
            self.width * self.dtype.size()
        } else {
            self.layout.tight_row_bytes()
        }
    }

    /// Row pitch of the CPU-side backing: the driver's staging pitch when
    /// there is staging, the texture's own row otherwise.
    fn backing_pitch(&self) -> usize {
        self.staging
            .as_ref()
            .map_or(self.layout.tight_row_bytes(), |s| s.row_pitch)
    }

    /// Bytes addressable from a map or a pin of this backing, padding
    /// included. Offsets, view bounds and `capacity_bytes` are in this space.
    fn backing_bytes(&self) -> usize {
        self.backing_pitch() * self.layout.mapped_rows()
    }

    /// The persistent host buffer, allocated on first use and shared by every
    /// view of this tensor.
    fn host_buffer(&self) -> &HostBuffer {
        self.host
            .get_or_init(|| HostBuffer::new(self.backing_pitch(), self.layout.mapped_rows()))
    }

    /// Orders the reads queued after this call behind the GPU write
    /// `set_gpu_write` recorded, with a GPU-side `Wait` on this device's
    /// fence.
    ///
    /// Does nothing when no write is recorded, when the device's fence is not
    /// the shared one ([`D3d11Device::fence_is_shared`]), when the value is
    /// above [`D3d11Device::last_signalled`], when the fence has passed the
    /// value, or when the device cannot wait.
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
        // SAFETY: both interfaces are live for the process's lifetime, and the
        // value names a point on that fence's timeline.
        if let Err(e) = unsafe { ctx4.Wait(fence, value) } {
            log::warn!(
                "D3D11 map: queuing a wait on the recorded write {value} failed ({e}); \
                 the staging refresh may read the texture while the GPU still writes it"
            );
        }
    }

    /// Refreshes the pinned host buffer from the texture, through the staging
    /// copy. A no-op for a tensor with no host buffer or no staging.
    ///
    /// Fails while a map of the staging copy is live.
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
        for row in 0..self.layout.mapped_rows() {
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
        if let Some(repack) = self.chroma_repack(buffer.row_pitch) {
            // SAFETY: the buffer holds every mapped row at its pitch.
            unsafe { repack.move_to_image(base) };
        }
        Ok(())
    }

    /// Publishes CPU writes to the texture from the backing that carries them:
    /// the host buffer once a writable pin has been taken on it, otherwise the
    /// staging copy. A no-op when the tensor has neither.
    ///
    /// Fails while a map of either backing is live.
    pub(crate) fn sync_for_device(&self, access: CpuAccess) -> Result<()> {
        if !access.writes() {
            return Ok(());
        }
        // Both claims are held until the function returns.
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
        // Publishes from the backing the caller wrote through.
        match (self.wrote_host.load(Ordering::Acquire), self.host.get()) {
            // Publishes every row of the host buffer.
            (true, Some(buffer)) => {
                if let Some(repack) = self.chroma_repack(buffer.row_pitch) {
                    // SAFETY: the buffer holds every mapped row at its pitch,
                    // and both claims are held.
                    unsafe { repack.move_to_texture(buffer.span().0) };
                }
                upload_buffer_window(
                    self.device,
                    &self.texture,
                    buffer,
                    &self.layout,
                    0,
                    self.layout.mapped_rows(),
                )
            }
            _ => match &self.staging {
                // SAFETY: both textures are live and identically described,
                // and `_staging_claim` keeps the staging texture unmapped.
                Some(st) => unsafe { self.device.ctx().CopyResource(&self.texture, &st.tex) },
                None => return Ok(()),
            },
        }
        publish(self.device);
        Ok(())
    }

    /// A stable host address for this tensor's rows, valid until the pin drops
    /// and across any number of maps and frames. The address is into the
    /// storage's host buffer; `sync_for_cpu` and `sync_for_device` move bytes
    /// between it and the texture.
    pub(crate) fn host_pin<'a>(&self, access: CpuAccess) -> Result<crate::pin::HostPin<'a>> {
        self.require_access(self.access)?;
        self.check_view_offset()?;
        if access.writes() {
            self.wrote_host.store(true, Ordering::Release);
        }
        let (base, len) = self.host_buffer().span();
        // SAFETY: `check_view_offset` above bounds `view_offset` by
        // `backing_bytes()`, which is the length this buffer was allocated at.
        let at = unsafe { base.add(self.view_offset) };
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
        self.check_view_offset()?;
        self.refuse_partial_write(access)?;
        match self.staging.as_ref() {
            Some(st) => self.staging_pin(st, access, non_blocking),
            None => self.buffer_pin(access),
        }
    }

    /// Bytes this tensor's image occupies from the start of its window: every
    /// row but the last at the backing's pitch, plus the last row's own bytes.
    fn addressable_span(&self) -> usize {
        let tight_row = self.image_row_bytes();
        if tight_row == 0 {
            return 0;
        }
        let tight = self.shape.iter().product::<usize>() * self.dtype.size();
        let rows = tight.div_ceil(tight_row);
        if rows == 0 {
            return 0;
        }
        let last_row = tight - (rows - 1) * tight_row;
        (rows - 1) * self.backing_pitch() + last_row
    }

    /// Refuses a window that cannot hold this tensor's own image: fails when
    /// `view_offset` plus [`addressable_span`](Self::addressable_span)
    /// overflows or runs past the backing.
    fn check_view_offset(&self) -> Result<()> {
        let capacity = self.backing_bytes();
        let end = self.view_offset.checked_add(self.addressable_span());
        match end {
            Some(end) if end <= capacity => Ok(()),
            _ => Err(Error::InsufficientCapacity {
                needed: end.unwrap_or(usize::MAX),
                capacity,
            }),
        }
    }

    /// Moves this tensor's window to `offset` bytes from the backing's origin,
    /// in the pitched space `view` measures in. Unchecked; `check_view_offset`
    /// refuses an offset past the backing when a pin is taken.
    pub(crate) fn set_view_offset(&mut self, offset: usize) {
        self.view_offset = offset;
    }

    /// This window as `(start_row, row_count)` in the backing's own texel
    /// rows, or `None` when `view_offset` is not a whole number of rows.
    fn row_window(&self) -> Option<(usize, usize)> {
        let pitch = self.backing_pitch();
        if pitch == 0 || !self.view_offset.is_multiple_of(pitch) {
            return None;
        }
        Some((self.view_offset / pitch, self.pitched_extent() / pitch))
    }

    /// Refuses a write-only map of a window shorter than the backing on a
    /// tensor with staging, with [`Error::PartialWriteRequiresReadWrite`].
    /// Returns `Ok` for every other access and for a tensor without staging.
    fn refuse_partial_write(&self, access: CpuAccess) -> Result<()> {
        if access != CpuAccess::Write || self.staging.is_none() {
            return Ok(());
        }
        let backing = self.backing_bytes();
        if self.view_offset == 0 && self.pitched_extent() >= backing {
            return Ok(());
        }
        Err(Error::PartialWriteRequiresReadWrite(format!(
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
        // The claim and the map run under one lock.
        let base = {
            let mut state = st.state();
            match state.claim(writes)? {
                // Joined a live read mapping.
                Some(base) => base,
                None => self.establish_map(st, &mut state, access, non_blocking)?,
            }
        };
        let guard = StagingGuard {
            device: self.device,
            texture: self.texture.clone(),
            staging: Arc::clone(st),
            writer: writes,
            repack: self.chroma_repack(st.row_pitch).filter(|_| writes),
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
        // A non-blocking map skips the refresh while one is pending; a
        // blocking map always refreshes.
        if access.reads() && !(non_blocking && state.refresh_pending) {
            self.wait_for_recorded_write();
            // SAFETY: both textures are live, identically described, and the
            // staging texture is unmapped (this caller holds the only claim).
            unsafe { self.device.ctx().CopyResource(&st.tex, &self.texture) };
            state.refresh_pending = true;
            if non_blocking {
                // Submits the copy.
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
                // `refresh_pending` stays set for the retry.
                return Err(Error::IoError(std::io::Error::from(
                    std::io::ErrorKind::WouldBlock,
                )));
            }
            return hr("ID3D11DeviceContext::Map", Err(e));
        }
        // The map consumed the queued copy.
        state.refresh_pending = false;
        state.base = mapped.pData as usize;
        if access.reads() {
            if let Some(repack) = self.chroma_repack(st.row_pitch) {
                // SAFETY: the mapping is `st.bytes()` long, which covers the
                // texture's every mapped row at `st.row_pitch`.
                unsafe { repack.move_to_image(mapped.pData.cast()) };
            }
        }
        Ok(state.base)
    }

    fn buffer_pin<'a>(&self, access: CpuAccess) -> Result<crate::pin::HostPin<'a>> {
        // Only a writable map publishes, and only the rows of its own window.
        let write_region = if access.writes() {
            // A native NV12 texture is published whole: a writable window
            // starts at its origin.
            let window = if self.layout.is_native_nv12() {
                if self.view_offset != 0 {
                    return Err(Error::InvalidArgument(format!(
                        "writable map of a native NV12 D3D11 texture tensor window ({} bytes                          at offset {}): a write-only NV12 texture is published whole, so map                          the tensor from its origin, or allocate it with CpuAccess::ReadWrite",
                        self.pitched_extent(),
                        self.view_offset
                    )));
                }
                Some((0, self.layout.mapped_rows()))
            } else {
                self.row_window()
            };
            Some(window.ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "writable map of a D3D11 texture tensor window ({} bytes at offset {}) not \
                     aligned to the {}-byte backing pitch: the host-buffer path can only \
                     publish whole rows",
                    self.pitched_extent(),
                    self.view_offset,
                    self.backing_pitch()
                ))
            })?)
        } else {
            None
        };
        let buffer = self.host_buffer();
        buffer.state().claim(access.writes())?;
        let (base, len) = buffer.span();
        let guard = HostBufferGuard {
            device: self.device,
            texture: self.texture.clone(),
            host: Arc::clone(&self.host),
            layout: self.layout,
            repack: self
                .chroma_repack(buffer.row_pitch)
                .filter(|_| write_region.is_some()),
            writeback: write_region,
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
        // SAFETY: `pin_impl` ran `check_view_offset`, which bounds
        // `view_offset` by `backing_bytes()`, the length of both backings.
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
    /// pitch, row padding included. A shape covering part of a row maps to the
    /// end of that row. The row count is the shape's byte count, taken with
    /// `dtype`, over the image's tight row.
    fn pitched_extent(&self) -> usize {
        let tight_row = self.image_row_bytes();
        if tight_row == 0 {
            return 0;
        }
        let tight = self.shape.iter().product::<usize>() * self.dtype.size();
        tight.div_ceil(tight_row) * self.backing_pitch()
    }

    /// Wraps a pin as a view of this tensor's shape. `byte_size_override` is
    /// `None` for an ordinary map, which exposes
    /// [`pitched_extent`](Self::pitched_extent), and the padded extent for
    /// [`map_with_byte_size`](Self::map_with_byte_size).
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

    /// Maps `bytes` of the pitched backing: the padded-row path.
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

    /// Whether the texture carries `D3D11_BIND_RENDER_TARGET`.
    #[allow(dead_code)]
    pub(crate) fn is_render_target(&self) -> bool {
        self.desc_bind_flags & D3D11_BIND_RENDER_TARGET.0 as u32 != 0
    }

    /// A duplicate of the texture's shared NT handle, owned by the caller.
    pub(crate) fn shared_handle(&self) -> Result<OwnedHandle> {
        duplicate_handle(&self.nt_handle).map_err(Error::IoError)
    }

    /// The tensor's own NT handle value, valid while the tensor lives.
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
    /// Monotonic: an older value never displaces a newer one. Any `u64` is
    /// accepted.
    pub(crate) fn set_gpu_write(&self, value: u64) {
        self.last_gpu_write.fetch_max(value, Ordering::AcqRel);
    }

    pub(crate) fn last_gpu_write(&self) -> u64 {
        self.last_gpu_write.load(Ordering::Acquire)
    }

    /// The row pitch a CPU map or pin sees, or `None` when there is none to
    /// report: the staging pitch when there is staging, the texture's width
    /// for a semi-planar tensor without staging, and `None` otherwise.
    pub(crate) fn image_backing_row_stride(&self) -> Option<usize> {
        if let Some(st) = self.staging.as_ref() {
            return Some(st.row_pitch);
        }
        // R8: the texture's row in bytes is its width in texels.
        (self.format.layout() == PixelLayout::SemiPlanar).then(|| self.layout.tight_row_bytes())
    }

    /// Bytes the image occupies with no row padding: the extent
    /// `set_logical_shape` measures against.
    pub(crate) fn tight_bytes(&self) -> usize {
        self.image_row_bytes() * self.layout.mapped_rows()
    }

    /// Imports this texture into CUDA. Returns `None` when no CUDA runtime
    /// loaded, when the adapter has no CUDA device, or when the driver refuses
    /// the import.
    ///
    /// The texture's NT handle is borrowed for the call; the fence's is a
    /// duplicate the CUDA handle takes over.
    pub(crate) fn try_init_cuda(&self) -> Option<crate::cuda::CudaHandle> {
        if !crate::cuda::is_cuda_available() {
            return None;
        }
        // Without a fence the import copies without waiting.
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

/// Submits the whole immediate context, which makes queued writes visible to
/// other devices that opened a texture's shared handle.
fn publish(device: &D3d11Device) {
    // SAFETY: the immediate context is live for the process's lifetime.
    unsafe { device.ctx().Flush() };
}

/// Uploads `row_count` of the host buffer's own rows into the texture at
/// `row_offset`, at the pitch the buffer was allocated with, through a boxed
/// `UpdateSubresource` spanning whole rows.
///
/// A native NV12 texture is uploaded whole, with no box.
fn upload_buffer_window(
    device: &D3d11Device,
    texture: &ID3D11Texture2D,
    buffer: &HostBuffer,
    layout: &D3d11ImageLayout,
    row_offset: usize,
    row_count: usize,
) {
    let (base, _) = buffer.span();
    if layout.is_native_nv12() {
        // SAFETY: the buffer holds every mapped row at `row_pitch`, and both
        // objects are live for the call.
        unsafe {
            device.ctx().UpdateSubresource(
                texture,
                0,
                None,
                base.cast_const().cast(),
                buffer.row_pitch as u32,
                (buffer.row_pitch * layout.mapped_rows()) as u32,
            );
        }
        return;
    }
    let texture_width = layout.texture_width;
    // SAFETY: `row_offset` rows are within the buffer's span.
    let src = unsafe { base.add(row_offset * buffer.row_pitch) };
    let region = D3D11_BOX {
        left: 0,
        top: row_offset as u32,
        front: 0,
        right: texture_width as u32,
        bottom: (row_offset + row_count) as u32,
        back: 1,
    };
    // SAFETY: the buffer and the destination texture are live, `region` is
    // within the texture's extent, and `src`'s rows are laid out at
    // `row_pitch`.
    unsafe {
        device.ctx().UpdateSubresource(
            texture,
            0,
            Some(&region),
            src.cast(),
            buffer.row_pitch as u32,
            0,
        );
    }
}

/// Releases one claim on the staging mapping, unmapping when it was the last
/// and, for the writer, publishing the rows to the texture.
struct StagingGuard {
    device: &'static D3d11Device,
    texture: ID3D11Texture2D,
    staging: Arc<Staging>,
    writer: bool,
    /// For a writer of a shorter image in a native NV12 texture: the chroma
    /// rows to move back to the texture's plane before the copy out.
    repack: Option<ChromaRepack>,
}

impl Drop for StagingGuard {
    fn drop(&mut self) {
        let mut state = self.staging.state();
        let base = state.base;
        if !state.release(self.writer) {
            // Another reader still holds the mapping.
            return;
        }
        if let Some(repack) = self.repack {
            // SAFETY: a writer holds the mapping alone and it is still
            // mapped: `base` addresses `staging.bytes()`, every mapped row.
            unsafe { repack.move_to_texture(base as *mut u8) };
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

/// Releases one claim on the host buffer and, for a writable map, uploads the
/// rows it wrote. `writeback` is `Some((row_offset, row_count))` for a
/// writable map and `None` for a read map.
struct HostBufferGuard {
    device: &'static D3d11Device,
    texture: ID3D11Texture2D,
    host: Arc<OnceLock<HostBuffer>>,
    layout: D3d11ImageLayout,
    /// For a writer of a shorter image in a native NV12 texture: the chroma
    /// rows to move back to the texture's plane before the upload.
    repack: Option<ChromaRepack>,
    writeback: Option<(usize, usize)>,
}

impl Drop for HostBufferGuard {
    fn drop(&mut self) {
        let buffer = self
            .host
            .get()
            .expect("the guard was built from a live host buffer");
        // The claim is held across the upload.
        let mut state = buffer.state();
        if let Some((row_offset, row_count)) = self.writeback {
            if let Some(repack) = self.repack {
                // SAFETY: the writer's claim is held, and the buffer holds
                // every mapped row at its pitch.
                unsafe { repack.move_to_texture(buffer.span().0) };
            }
            upload_buffer_window(
                self.device,
                &self.texture,
                buffer,
                &self.layout,
                row_offset,
                row_count,
            );
            publish(self.device);
        }
        state.release(self.writeback.is_some());
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
        // Texture storage reports the shared `DmaBuf` variant.
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

    /// Reports `WouldBlock` while the staging refresh copy is in flight.
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
        // Measured against the image's unpadded size.
        let needed = shape.iter().product::<usize>() * std::mem::size_of::<T>();
        let capacity = self.tight_bytes();
        if needed > capacity {
            return Err(Error::InsufficientCapacity { needed, capacity });
        }
        // A native NV12 texture holds a smaller NV12 image as its top-left
        // corner, with the image's chroma at the top of the texture's chroma
        // plane. Only an NV12 shape is accepted.
        if self.layout.is_native_nv12() {
            let rows = match shape {
                [rows, _] => PixelFormat::Nv12.image_height_from_combined(*rows),
                _ => None,
            };
            if rows.is_none_or(|h| h > self.layout.texture_height) {
                return Err(Error::InvalidArgument(format!(
                    "a native NV12 D3D11 texture {}x{} holds an NV12 image of at most that \
                     size, not a {shape:?} shape",
                    self.layout.texture_width, self.layout.texture_height
                )));
            }
        }
        self.shape = shape.to_vec();
        Ok(())
    }

    /// Zero-copy sub-region view sharing this texture, its staging, its host
    /// buffer, its NT handle and its [`BufferIdentity`], positioned at
    /// `offset_bytes` from this tensor's own window.
    ///
    /// `offset_bytes` is in backing (pitched) space: whole rows are
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
        // SAFETY: both out-parameters are valid locals.
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

    /// Uploads `churn` zero fills and then `byte` to the texture `handle`
    /// names, from `other`, then signals `fence` at `value` and submits.
    #[allow(clippy::too_many_arguments)]
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
        // SAFETY: the second device's immediate context is live.
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

    /// An RGBA8 tensor of `w` x `h` with `access`.
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

    /// A blocking map after an abandoned `try_map` queues a fresh refresh copy.
    #[test]
    fn a_blocking_map_refreshes_past_an_abandoned_try_maps_copy() {
        let d = device().unwrap();
        let old = rgba(64, 64, CpuAccess::ReadWrite);
        let new = rgba(64, 64, CpuAccess::ReadWrite);
        for (t, byte) in [(&old, 0x11u8), (&new, 0x22u8)] {
            let mut m = t.map_with(CpuAccess::Write).unwrap();
            m.as_mut_slice().fill(byte);
        }

        // Builds the state an abandoned `try_map` leaves: one refresh copy
        // queued and `refresh_pending` set.
        {
            let st = old.staging.as_ref().unwrap();
            let mut state = st.state();
            // SAFETY: both textures are live and identically described, and
            // the state lock is held, so the staging texture stays unmapped.
            unsafe { d.ctx().CopyResource(&st.tex, &old.texture) };
            state.refresh_pending = true;
        }

        // A GPU write issued after the queued refresh copy.
        // SAFETY: both textures are live and identically described.
        unsafe { d.ctx().CopyResource(&old.texture, &new.texture) };
        d.signal();

        // Compares only the logical bytes of each row.
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

    /// A write-only map of a row window publishes only that window's rows.
    #[test]
    fn write_only_partial_row_window_does_not_clobber_rows_outside_it() {
        let d = device().unwrap();
        const W: usize = 64;
        const H: usize = 12;
        const WINDOW_START_ROW: usize = 4;
        const WINDOW_ROWS: usize = 4;

        let pool = rgba(W, H, CpuAccess::Write);
        {
            // Seeds every row of the texture with a sentinel.
            let mut m = pool.map_with(CpuAccess::Write).unwrap();
            m.as_mut_slice().fill(0xAA);
        }
        {
            let offset = WINDOW_START_ROW * pool.backing_pitch();
            let shape = vec![WINDOW_ROWS, W, 4];
            let window = pool
                .view(offset, &shape)
                .expect("a CpuAccess::Write tensor's own row-aligned sub-window must be viewable");
            let mut m = window
                .map_with(CpuAccess::Write)
                .expect("a row-aligned partial window must be mappable for write");
            m.as_mut_slice().fill(0x55);
        }

        pool.sync_for_device(CpuAccess::Write).unwrap();

        // Reads the texture back through a second, independently allocated
        // tensor.
        let readback = rgba(W, H, CpuAccess::ReadWrite);
        // SAFETY: both textures are live and identically described.
        unsafe { d.ctx().CopyResource(&readback.texture, &pool.texture) };
        let mapped = readback.map_with(CpuAccess::Read).unwrap();
        let pitch = readback.backing_pitch();
        let row_bytes = readback.layout().tight_row_bytes();
        let bytes = mapped.as_slice();
        let logical = |r: usize| &bytes[r * pitch..r * pitch + row_bytes];

        for r in 0..H {
            let expect = if (WINDOW_START_ROW..WINDOW_START_ROW + WINDOW_ROWS).contains(&r) {
                0x55
            } else {
                0xAA
            };
            assert!(
                logical(r).iter().all(|&b| b == expect),
                "row {r}: expected every byte == {expect:#x}, the windowed write touched a row \
                 outside its own range"
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

    /// Combined-plane sizes for the stride tests: NV16 and NV24 at an even
    /// size, and NV12 at odd sizes.
    fn nv_sizes() -> [(PixelFormat, usize, usize); 4] {
        [
            (PixelFormat::Nv12, 64, 63),
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

    /// A semi-planar texture is exactly as wide as the row pitch a CPU map sees.
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

            // The public constructor reports that width as its row stride, or
            // no stride when the driver added no padding.
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

    /// A semi-planar tensor with no staging is as wide as, and reports, the driver's pitch.
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

    /// A write-only NV tensor's chroma lines land at the texture's pitch.
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

        // Chroma line `i` is `2 * w` bytes at combined-plane row `h + 2 * i`.
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

        // A second, readable wrapper of the same texture.
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

    /// An external semi-planar texture is wrapped at its own width, which becomes the stride.
    #[test]
    fn an_external_semi_planar_texture_is_wrapped_at_its_own_width() {
        let backing = nv_tensor(PixelFormat::Nv16, 128, 64, CpuAccess::None);
        assert_eq!(backing.layout().texture_width, 128);
        let shape = PixelFormat::Nv16.allocation_shape(100, 64).unwrap();
        // SAFETY: `backing` holds the texture live across this call.
        let t = unsafe {
            D3d11TextureTensor::<u8>::from_d3d11_texture(
                backing.texture_ptr(),
                100,
                64,
                PixelFormat::Nv16,
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

    /// An external semi-planar texture narrower than its own staging pitch is refused.
    #[test]
    fn an_external_semi_planar_texture_narrower_than_its_pitch_is_refused() {
        let (w, h) = (100usize, 64usize);
        let fmt = PixelFormat::Nv12;
        let d = device().unwrap();
        // The combined plane at exactly the image's own width.
        let raw = D3d11ImageLayout {
            texture_width: w,
            ..combined_semi_planar_layout(fmt, w, h).unwrap()
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

    /// A wrapped semi-planar tensor reports the texture's width as its row stride.
    #[test]
    fn wrapped_nv_tensors_record_the_textures_width_as_their_row_stride() {
        for (w, h) in [(321usize, 241usize), (64, 63)] {
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

    /// A native NV12 texture round-trips luma and chroma through combined-plane CPU maps.
    #[test]
    fn native_nv12_round_trips_luma_and_chroma_through_the_cpu() {
        let (w, h) = (640usize, 480usize);
        let fmt = PixelFormat::Nv12;
        for access in [CpuAccess::ReadWrite, CpuAccess::Write] {
            let t =
                crate::Tensor::<u8>::image(w, h, fmt, Some(crate::TensorMemory::DmaBuf), access)
                    .unwrap();
            let layout = t.d3d11_layout().unwrap();
            assert!(layout.is_native_nv12(), "{access:?}: {layout:?}");
            assert_eq!(
                (layout.texture_width, layout.texture_height),
                (w, h),
                "{access:?}: the texture is the luma grid"
            );
            let stride = t.effective_row_stride().unwrap();
            assert!(stride >= w, "{access:?}: stride {stride}");
            let byte = |r: usize, x: usize| ((r * 7 + x * 3) % 253) as u8;
            {
                let mut m = t.map_write().unwrap();
                let dst = m.as_mut_slice();
                for r in 0..h + h / 2 {
                    for x in 0..w {
                        dst[r * stride + x] = byte(r, x);
                    }
                }
            }
            let shape = fmt.allocation_shape(w, h).unwrap();
            // SAFETY: `t` holds the texture alive across this call.
            let r = unsafe {
                D3d11TextureTensor::<u8>::from_d3d11_texture(
                    t.d3d11_texture().unwrap(),
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
            let rstride = r.image_backing_row_stride().unwrap();
            let m = r.map_with(CpuAccess::Read).unwrap();
            let src = m.as_slice();
            for row in 0..h + h / 2 {
                for x in 0..w {
                    assert_eq!(
                        src[row * rstride + x],
                        byte(row, x),
                        "{access:?}: row {row} column {x}"
                    );
                }
            }
        }
    }

    /// An external native NV12 texture wraps by pointer and by shared handle.
    #[test]
    fn an_external_native_nv12_texture_wraps_by_pointer_and_by_handle() {
        let (w, h) = (1920usize, 1080usize);
        let fmt = PixelFormat::Nv12;
        let d = device().unwrap();
        let layout = image_d3d11_layout(fmt, DType::U8, w, h).unwrap();
        assert!(layout.is_native_nv12());
        let tex = create_texture(
            d,
            &texture_desc(&layout, D3D11_USAGE_DEFAULT, BIND, 0, MISC),
        )
        .unwrap();

        // SAFETY: `tex` is live for the whole call.
        let by_ptr = unsafe {
            crate::Tensor::<u8>::from_d3d11_texture(tex.as_raw(), w, h, fmt, CpuAccess::Read, None)
        }
        .unwrap();
        assert!(by_ptr.d3d11_layout().unwrap().is_native_nv12());
        assert_eq!((by_ptr.width(), by_ptr.height()), (Some(w), Some(h)));
        // SAFETY: `tex` is live for the call.
        let geometry = unsafe {
            crate::d3d11::d3d11_texture_geometry(tex.as_raw(), fmt, Some(&[h * 3 / 2, w]))
        }
        .unwrap();
        assert_eq!(geometry, (w, h));

        let handle = create_nt_handle(&tex).unwrap();
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
        .unwrap();
        assert!(by_handle.d3d11_layout().unwrap().is_native_nv12());
        // SAFETY: `handle` is valid for the call.
        let geometry = unsafe {
            crate::d3d11::d3d11_shared_handle_geometry(handle.as_raw_handle(), fmt, None)
        }
        .unwrap();
        assert_eq!(geometry, (w, h), "an NV12 texture needs no shape");

        // Wrapping at another image size is refused.
        // SAFETY: `tex` is live for the call.
        let other = unsafe {
            crate::Tensor::<u8>::from_d3d11_texture(
                tex.as_raw(),
                w,
                h - 2,
                fmt,
                CpuAccess::Read,
                None,
            )
        };
        assert!(matches!(other, Err(Error::InvalidArgument(_))), "{other:?}");
    }

    /// A shorter image in a native NV12 texture round-trips its chroma through CPU views.
    #[test]
    fn a_shorter_image_in_a_native_nv12_texture_keeps_its_chroma_in_the_plane() {
        let (w, h) = (640usize, 480usize);
        let (iw, ih) = (320usize, 240usize);
        for access in [CpuAccess::ReadWrite, CpuAccess::Write] {
            let mut t = crate::Tensor::<u8>::image(
                w,
                h,
                PixelFormat::Nv12,
                Some(crate::TensorMemory::DmaBuf),
                access,
            )
            .unwrap();
            assert!(t.d3d11_layout().unwrap().is_native_nv12());
            t.configure_image(iw, ih, PixelFormat::Nv12).unwrap();
            assert_eq!((t.width(), t.height()), (Some(iw), Some(ih)));
            let stride = t.effective_row_stride().unwrap();
            let byte = |r: usize, x: usize| ((r * 11 + x * 5) % 251) as u8;
            {
                let mut m = t.map_for_write().unwrap();
                let dst = m.as_mut_slice();
                for r in 0..ih + ih / 2 {
                    for x in 0..iw {
                        dst[r * stride + x] = byte(r, x);
                    }
                }
            }
            if access == CpuAccess::ReadWrite {
                let m = t.map_read().unwrap();
                for r in 0..ih + ih / 2 {
                    for x in 0..iw {
                        assert_eq!(m.as_slice()[r * stride + x], byte(r, x), "row {r}");
                    }
                }
            }
            let shape = PixelFormat::Nv12.allocation_shape(w, h).unwrap();
            // SAFETY: `t` holds the texture alive across this call.
            let whole = unsafe {
                D3d11TextureTensor::<u8>::from_d3d11_texture(
                    t.d3d11_texture().unwrap(),
                    w,
                    h,
                    PixelFormat::Nv12,
                    DType::U8,
                    &shape,
                    CpuAccess::Read,
                    None,
                )
            }
            .unwrap();
            let pitch = whole.image_backing_row_stride().unwrap();
            let m = whole.map_with(CpuAccess::Read).unwrap();
            for r in 0..ih {
                for x in 0..iw {
                    assert_eq!(
                        m.as_slice()[r * pitch + x],
                        byte(r, x),
                        "{access:?} luma {r}"
                    );
                }
            }
            for r in 0..ih / 2 {
                for x in 0..iw {
                    assert_eq!(
                        m.as_slice()[(h + r) * pitch + x],
                        byte(ih + r, x),
                        "{access:?} chroma {r} at the top of the texture's plane"
                    );
                }
            }
        }
        let mut t = crate::Tensor::<u8>::image(
            w,
            h,
            PixelFormat::Nv12,
            Some(crate::TensorMemory::DmaBuf),
            CpuAccess::ReadWrite,
        )
        .unwrap();
        assert!(
            t.configure_image(w, h + 2, PixelFormat::Nv12).is_err(),
            "taller"
        );
        assert!(
            t.configure_image(w, h, PixelFormat::Rgba).is_err(),
            "not NV12"
        );
    }

    /// A writable map of part of a write-only native NV12 texture is refused.
    #[test]
    fn a_partial_write_window_of_a_write_only_native_nv12_tensor_is_refused() {
        let (w, h) = (64usize, 64usize);
        let fmt = PixelFormat::Nv12;
        let shape = fmt.allocation_shape(w, h).unwrap();
        let t = D3d11TextureTensor::<u8>::new_image(
            w,
            h,
            fmt,
            DType::U8,
            &shape,
            None,
            CpuAccess::Write,
        )
        .unwrap();
        assert!(t.layout().is_native_nv12());
        let pitch = t.image_backing_row_stride().unwrap();
        let view = t.view(pitch * 8, &[8, w]).unwrap();
        let err = view.map_with(CpuAccess::Write).unwrap_err();
        assert!(
            matches!(err, Error::InvalidArgument(_)) && err.to_string().contains("NV12"),
            "{err}"
        );
        assert!(
            t.map_with(CpuAccess::Write).is_ok(),
            "the whole tensor maps"
        );
    }

    /// An external semi-planar texture narrower than an even image row is refused.
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
        // Reads it back through a second, readable wrapper of the same
        // texture.
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
        // Waits on the shared fence for the write to land.
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
        // A second D3D11 device writes the texture through its shared handle
        // and signals a fence of its own behind that write.
        assert!(
            device().unwrap().fence_is_shared(),
            "this host's copy signals the process's shared fence, so the map \
             path's wait is live rather than skipped"
        );
        // Sixteen zero-fill uploads precede the 1 MiB write.
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
        let t = rgba(32, 32, CpuAccess::ReadWrite);
        {
            let mut v = t.map_with(CpuAccess::Write).unwrap();
            v.as_mut_slice().fill(0x77);
        }
        let unreachable = device().unwrap().last_signalled() + 1_000_000;
        t.set_gpu_write(unreachable);

        // Polls `try_map_with` until a deadline, yielding between attempts.
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
        // The recorded value is still reported.
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
        // The import records a newer value on the local fence, not the
        // producer's.
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

    /// A host pin's address is 64-byte aligned.
    #[test]
    fn a_host_pin_is_64_byte_aligned() {
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

    /// A map covers this tensor's own rows at the backing pitch, not the rest of the parent.
    #[test]
    fn a_map_covers_this_tensors_rows_not_the_parents_tail() {
        let t = rgba(32, 8, CpuAccess::ReadWrite);
        let pitch = t.image_backing_row_stride().unwrap();

        let whole = t.map_with(CpuAccess::Read).unwrap();
        assert_eq!(whole.as_slice().len(), pitch * 8);
        drop(whole);

        // A window two rows in, two rows tall.
        let v = t.view(pitch * 2, &[2, 32, 4]).unwrap();
        let mapped = v.map_with(CpuAccess::Read).unwrap();
        assert_eq!(mapped.as_slice().len(), pitch * 2);
        drop(mapped);

        // An explicit request past the window is refused.
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

        // The first attempt either maps or reports `WouldBlock`.
        match t.try_map_with(CpuAccess::Read) {
            Ok(_) => {}
            Err(Error::IoError(e)) if e.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(e) => panic!("first try_map_with returned an unexpected error: {e}"),
        }

        // Retries until a deadline, yielding between attempts.
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
        // A readable wrapper of the same texture.
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

        // A read map of the write-only tensor publishes nothing.
        drop(w.map_with(CpuAccess::Read).unwrap());

        {
            let mut v = r.map_with(CpuAccess::Write).unwrap();
            let dst = v.as_mut_slice();
            for n in 0..32 {
                dst[n * pitch..n * pitch + row].fill(0x11);
            }
        }
        // A second read map, after the overwrite.
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
        // Three whole rows in backing space.
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
