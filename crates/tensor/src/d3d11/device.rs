// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//! The process-wide D3D11 device every texture tensor lives on and ANGLE
//! renders with, plus the shared fence that orders GPU consumers.
//!
//! One process gets one device. Independently linked copies of this crate
//! find each other through a named file mapping (the rendezvous) keyed on
//! the process id, so a texture allocated by one copy is usable by the
//! others. The rendezvous carries the device pointer, the fence's shared NT
//! handle and the signal counter, so every copy signals one fence from one
//! sequence of values. A host that already owns a device installs it with
//! [`use_external_device`] before the first [`device`] call.

use super::adapter::{select_adapter, AdapterSelection};
use super::com;
use std::ffi::c_void;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle, RawHandle};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use windows::core::{IUnknown, Interface, PCWSTR};
use windows::Win32::Foundation::{
    CloseHandle, DuplicateHandle, DUPLICATE_SAME_ACCESS, GENERIC_ALL, HANDLE, HMODULE,
    INVALID_HANDLE_VALUE, WAIT_OBJECT_0,
};
use windows::Win32::Graphics::Direct3D::{
    D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_UNKNOWN, D3D_DRIVER_TYPE_WARP, D3D_FEATURE_LEVEL,
    D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_12_0, D3D_FEATURE_LEVEL_12_1,
};
use windows::Win32::Graphics::Direct3D11::{
    D3D11CreateDevice, ID3D11Device, ID3D11Device1, ID3D11Device5, ID3D11DeviceContext,
    ID3D11DeviceContext4, ID3D11Fence, ID3D11Multithread, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
    D3D11_CREATE_DEVICE_DEBUG, D3D11_CREATE_DEVICE_FLAG, D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
    D3D11_FENCE_FLAG_SHARED, D3D11_SDK_VERSION,
};
use windows::Win32::Graphics::Dxgi::{IDXGIAdapter, IDXGIDevice};
use windows::Win32::System::Memory::{
    CreateFileMappingW, MapViewOfFile, OpenFileMappingW, UnmapViewOfFile, FILE_MAP_READ,
    FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, PAGE_READWRITE,
};
use windows::Win32::System::Threading::{
    CreateEventW, GetCurrentProcess, GetCurrentProcessId, WaitForSingleObject,
};

/// Layout version of the rendezvous mapping. A copy of this crate built
/// against a different layout reads the mismatch and creates its own device
/// instead of trusting the bytes.
const RENDEZVOUS_VERSION: u64 = 2;
/// Written into the version word between claiming the record and filling it
/// in. A reader that sees it waits for the claim holder instead of reading
/// the half-written record behind it.
const RENDEZVOUS_CLAIMING: u64 = u64::MAX;
/// Turns a reader or a would-be publisher takes while another copy's claim is
/// in flight. The claim holder writes two words between its CAS and its
/// release store, so this only has to outlast a preemption.
const RENDEZVOUS_CLAIM_SPINS: u32 = 1024;
/// How many of those turns yield before the rest start sleeping. The common
/// wait is a few stores by a runnable thread, which a yield covers; the budget
/// past that is for a claim holder the scheduler has taken off a core.
const RENDEZVOUS_CLAIM_YIELDS: u32 = 64;
/// How long each turn past [`RENDEZVOUS_CLAIM_YIELDS`] sleeps, making the
/// whole budget roughly 50 ms rather than a few microseconds of spinning.
const RENDEZVOUS_CLAIM_SLEEP_US: u64 = 50;
/// Bytes in the rendezvous mapping:
/// `[version: u64, device_ptr: u64, fence_handle_value: u64, signal_counter: u64]`.
const RENDEZVOUS_BYTES: usize = 32;
/// Byte offset of the shared signal counter inside the mapping. The view is
/// page-aligned, so the `AtomicU64` there is aligned.
const COUNTER_OFFSET: usize = 24;
/// The DXGI description WARP reports.
const WARP_DESCRIPTION: &str = "Microsoft Basic Render Driver";
/// Label recorded when this copy adopts a device another copy published.
const ADOPTED_LABEL: &str = "adopted from another copy in this process";

/// A point in the shared fence's timeline plus a duplicate of the fence's
/// shared NT handle, so a consumer in another API can wait for the GPU work
/// a tensor's producer queued.
pub struct GpuCompletion {
    pub fence: OwnedHandle,
    pub value: u64,
}

/// Where the newest allocated fence value is kept.
enum Counter {
    /// The word inside a rendezvous view this process keeps mapped for its
    /// lifetime, so every copy of this crate allocates from one sequence.
    Shared(&'static AtomicU64),
    /// No usable rendezvous: this copy counts on its own.
    Local(AtomicU64),
}

impl Counter {
    fn word(&self) -> &AtomicU64 {
        match self {
            Counter::Shared(word) => word,
            Counter::Local(word) => word,
        }
    }
}

/// The process device: the `ID3D11Device` and immediate context every
/// texture tensor is created on, with the shared fence used to order
/// consumers behind the work queued on that context.
///
/// The COM interfaces below are read only by the texture storage
/// (`super::texture`), which the `dynamic` backend does not compile -- it
/// forwards every tensor call to the C ABI instead. The device itself is still
/// built there, because the adapter and fence probes are part of the public
/// surface on both backends.
#[cfg_attr(not(feature = "static"), allow(dead_code))]
pub struct D3d11Device {
    dev: ID3D11Device,
    dev1: Option<ID3D11Device1>,
    dev5: Option<ID3D11Device5>,
    ctx: ID3D11DeviceContext,
    ctx4: Option<ID3D11DeviceContext4>,
    mt: Option<ID3D11Multithread>,
    fence: Option<ID3D11Fence>,
    fence_handle: Option<OwnedHandle>,
    adapter: Option<IDXGIAdapter>,
    counter: Counter,
    /// Orders signals when the device has no `ID3D11Multithread`. Only ever
    /// held inside `signal`.
    signal_lock: Mutex<()>,
    adapter_label: String,
    luid: (i32, u32),
    is_warp: bool,
    creation_flags: u32,
}

static DEVICE: OnceLock<std::result::Result<D3d11Device, String>> = OnceLock::new();
static EXTERNAL: Mutex<Option<usize>> = Mutex::new(None);

/// The process device, created on first call. A failed creation is cached
/// too, so a host that cannot make a device is told the same thing every
/// time instead of retrying on every allocation.
pub fn device() -> crate::Result<&'static D3d11Device> {
    DEVICE
        .get_or_init(|| create().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|s| crate::Error::IoError(std::io::Error::other(s.clone())))
}

/// Installs a host-owned device before the first `device()` call. It is
/// published through the rendezvous so every copy of this crate in the
/// process adopts it.
///
/// # Ordering
///
/// This must run before any call that reaches [`device()`], which caches the
/// process device on first use and returns
/// [`Error::InvalidOperation`](crate::Error::InvalidOperation) here once it
/// has. In practice that means before
/// [`is_gpu_buffer_available`](crate::is_gpu_buffer_available), before
/// `Tensor::image` with `TensorMemory::DmaBuf`, and before the image
/// processor -- each of which creates the device as a side effect of doing
/// its job.
///
/// "Already initialized" is a question about the *process*, not about this
/// copy of the crate: a copy whose sibling already created the device has an
/// empty `DEVICE` of its own but finds the device published in the
/// rendezvous, and installing an external one there would write a pointer
/// that no copy ever adopts -- `device()` prefers the published device, so
/// the host's would be silently ignored rather than refused. Both are
/// [`Error::InvalidOperation`](crate::Error::InvalidOperation).
///
/// # Safety
///
/// `ptr` must be a live `ID3D11Device*`. No reference is taken here, so the
/// caller must keep the device alive until the first [`device()`] call, which
/// clones the pointer and takes the reference this crate then holds.
pub unsafe fn use_external_device(ptr: *mut c_void) -> crate::Result<()> {
    if ptr.is_null() {
        return Err(crate::Error::InvalidArgument(
            "use_external_device: null device".into(),
        ));
    }
    if let Some(cached) = DEVICE.get() {
        return Err(crate::Error::InvalidOperation(match cached {
            // A cached failure is not "too late", it is "impossible here":
            // this process will never get a device, so a host that reads
            // "already initialized" would keep looking for the call it lost
            // the race to.
            Err(e) => format!(
                "use_external_device: D3D11 device creation already failed in this process \
                 and the failure is cached: {e}"
            ),
            Ok(_) => "use_external_device: the D3D11 device is already initialized".into(),
        }));
    }
    // The cross-copy half of the same question. Read after `DEVICE` because
    // this one maps and unmaps a file mapping, and the common case is a
    // second call in the copy that made the first.
    if read_rendezvous().is_some() {
        return Err(crate::Error::InvalidOperation(
            "use_external_device: the D3D11 device is already initialized by another \
             copy of this library in this process"
                .into(),
        ));
    }
    // Checked here rather than at adoption so a caller that passes the wrong
    // pointer is told at the call it can still fix. No reference is taken;
    // `adopt` clones on first use.
    if !device_pointer_is_live(ptr as usize) {
        return Err(crate::Error::InvalidArgument(
            "use_external_device: pointer is not a live ID3D11Device".into(),
        ));
    }
    *EXTERNAL.lock().unwrap_or_else(|e| e.into_inner()) = Some(ptr as usize);
    Ok(())
}

/// Takes a counted reference on a device pointer this process owns but this
/// copy of the crate does not hold a reference to yet.
fn adopt(ptr: usize) -> crate::Result<ID3D11Device> {
    // SAFETY: `ptr` is a live `ID3D11Device*` in this process -- either
    // published through the rendezvous by another copy of this crate, or
    // handed over by the host through `use_external_device`. Borrowing and
    // then cloning AddRefs without consuming the caller's reference.
    unsafe { ID3D11Device::from_raw_borrowed(&(ptr as *mut c_void)) }
        .cloned()
        .ok_or_else(|| crate::Error::InvalidArgument("D3D11 device pointer is null".into()))
}

fn create() -> crate::Result<D3d11Device> {
    if let Some(view) = open_rendezvous() {
        return adopt_published(view);
    }
    let external = *EXTERNAL.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(ptr) = external {
        return finish(adopt(ptr)?, "host-provided device".into(), None);
    }
    let selected = select_adapter();
    let warp = selected.selection == AdapterSelection::Warp;
    let debug = std::env::var("EDGEFIRST_D3D11_DEBUG")
        .map(|v| v == "1")
        .unwrap_or(false);
    let base = if debug {
        D3D11_CREATE_DEVICE_DEBUG
    } else {
        D3D11_CREATE_DEVICE_FLAG(0)
    };
    // Video support is what Media Foundation needs; WARP refuses it, so fall
    // back in order and record which flags took.
    let attempts = [
        base | D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
        base | D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        base,
    ];
    let levels = [
        D3D_FEATURE_LEVEL_12_1,
        D3D_FEATURE_LEVEL_12_0,
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
    ];
    let mut last = None;
    for flags in attempts {
        let mut dev: Option<ID3D11Device> = None;
        let mut fl = D3D_FEATURE_LEVEL::default();
        let adapter = selected
            .adapter
            .as_ref()
            .map(|a| &*a.adapter)
            .filter(|_| !warp);
        let driver = if warp {
            D3D_DRIVER_TYPE_WARP
        } else if adapter.is_some() {
            D3D_DRIVER_TYPE_UNKNOWN
        } else {
            D3D_DRIVER_TYPE_HARDWARE
        };
        // SAFETY: documented creation call; both out-parameters are valid
        // locals, and the adapter reference, when given, is a live
        // `IDXGIAdapter1` `selected` holds for the whole call.
        let result = unsafe {
            D3D11CreateDevice(
                adapter,
                driver,
                HMODULE::default(),
                flags,
                Some(&levels),
                D3D11_SDK_VERSION,
                Some(&mut dev),
                Some(&mut fl),
                None,
            )
        };
        match result {
            Ok(()) => {
                let dev = dev.ok_or_else(|| {
                    crate::Error::IoError(std::io::Error::other(
                        "D3D11CreateDevice returned no device",
                    ))
                })?;
                // Only an explicit WARP selection settles the question; an
                // adapter picked by LUID or description is left to detection,
                // which reads the DXGI description back off the device.
                let d = finish(dev, selected.label.clone(), warp.then_some(true))?;
                log::info!(
                    "D3D11 device up on {} (flags {:#x}, feature level {:#x})",
                    d.adapter_label,
                    flags.0,
                    fl.0
                );
                return Ok(d);
            }
            Err(e) => {
                log::debug!(
                    "D3D11CreateDevice(flags {:#x}) failed: {:#010x}; retrying with fewer flags",
                    flags.0,
                    e.code().0 as u32
                );
                last = Some(e);
            }
        }
    }
    com::hr(
        "D3D11CreateDevice",
        Err(last.expect("the attempt list is not empty")),
    )
}

/// Wraps a freshly created or host-provided device and publishes it, or
/// adopts the device another copy of this crate published first.
fn finish(dev: ID3D11Device, label: String, is_warp: Option<bool>) -> crate::Result<D3d11Device> {
    let mut d = wrap(dev, label, is_warp, None)?;
    match publish_rendezvous(d.raw() as usize, d.fence_handle_value()) {
        Published::Ours(counter) => {
            d.counter = Counter::Shared(counter);
            Ok(d)
        }
        Published::Other(view) => {
            log::warn!(
                "the rendezvous for this process already names D3D11 device {:#x}; dropping the wrapper \
                 just built for {:#x} and adopting the published device, fence and counter",
                view.device_ptr,
                d.raw() as usize
            );
            drop(d);
            adopt_published(view)
        }
        Published::Unavailable => Ok(d),
    }
}

/// Builds a wrapper on the device another copy of this crate published,
/// sharing its fence and its signal counter.
fn adopt_published(view: RendezvousView) -> crate::Result<D3d11Device> {
    let dev = adopt(view.device_ptr)?;
    wrap(dev, ADOPTED_LABEL.into(), None, Some(&view))
}

fn wrap(
    dev: ID3D11Device,
    adapter_label: String,
    is_warp: Option<bool>,
    published: Option<&RendezvousView>,
) -> crate::Result<D3d11Device> {
    let dev1 = dev.cast::<ID3D11Device1>().ok();
    let dev5 = dev.cast::<ID3D11Device5>().ok();
    // SAFETY: `dev` is live.
    let ctx = com::hr("ID3D11Device::GetImmediateContext", unsafe {
        dev.GetImmediateContext()
    })?;
    let ctx4 = ctx.cast::<ID3D11DeviceContext4>().ok();
    let mt = dev.cast::<ID3D11Multithread>().ok();
    // Non-GL threads copy, map and signal on this context while the GL
    // worker renders; the runtime's own lock makes each call atomic.
    if let Some(mt) = &mt {
        // SAFETY: `mt` is live. The return value is the previous setting.
        let _ = unsafe { mt.SetMultithreadProtected(true) };
    }
    let (fence, fence_handle, counter) = fence_and_counter(dev5.as_ref(), published)?;
    let (luid, is_warp_detected, adapter) = adapter_identity(&dev);
    // SAFETY: `dev` is live.
    let creation_flags = unsafe { dev.GetCreationFlags() };
    Ok(D3d11Device {
        dev,
        dev1,
        dev5,
        ctx,
        ctx4,
        mt,
        fence,
        fence_handle,
        adapter,
        counter,
        signal_lock: Mutex::new(()),
        adapter_label,
        luid,
        is_warp: is_warp.unwrap_or(is_warp_detected),
        creation_flags,
    })
}

/// The fence this copy signals and the counter it allocates values from.
///
/// When the rendezvous names a fence this device can open, both are the
/// published ones, so completions recorded by one copy mean the same thing
/// to every other. Otherwise this copy gets a fence and a counter of its
/// own, and says so.
fn fence_and_counter(
    dev5: Option<&ID3D11Device5>,
    published: Option<&RendezvousView>,
) -> crate::Result<(Option<ID3D11Fence>, Option<OwnedHandle>, Counter)> {
    let Some(d5) = dev5 else {
        if published.is_some() {
            log::warn!("no ID3D11Device5 on this device: the published fence cannot be opened and this copy has no fence at all");
        }
        return Ok((None, None, Counter::Local(AtomicU64::new(0))));
    };
    if let Some(view) = published {
        match open_published_fence(d5, view) {
            Ok((fence, handle)) => return Ok((Some(fence), Some(handle), Counter::Shared(view.counter))),
            Err(e) => log::warn!(
                "cannot open the fence published for this process ({e}); this copy signals a fence of its own \
                 and its completion values are not comparable with the other copies'"
            ),
        }
    }
    let mut f: Option<ID3D11Fence> = None;
    // SAFETY: `d5` is live and `f` is a valid out-parameter.
    com::hr("ID3D11Device5::CreateFence", unsafe {
        d5.CreateFence(0, D3D11_FENCE_FLAG_SHARED, &mut f)
    })?;
    let fence = f.ok_or_else(|| {
        crate::Error::IoError(std::io::Error::other("CreateFence returned no fence"))
    })?;
    // SAFETY: `fence` is live; the handle is unnamed and this process owns it.
    let h = com::hr("ID3D11Fence::CreateSharedHandle", unsafe {
        fence.CreateSharedHandle(None, GENERIC_ALL.0, PCWSTR::null())
    })?;
    // SAFETY: `h` is an NT handle this call just created and nothing else holds.
    Ok((
        Some(fence),
        Some(unsafe { OwnedHandle::from_raw_handle(h.0) }),
        Counter::Local(AtomicU64::new(0)),
    ))
}

/// Opens the fence the rendezvous names, keeping a duplicate of its handle
/// so this copy's `fence_shared_handle` and `fence_handle_value` behave the
/// same as the publisher's.
fn open_published_fence(
    d5: &ID3D11Device5,
    view: &RendezvousView,
) -> crate::Result<(ID3D11Fence, OwnedHandle)> {
    if view.fence_handle == 0 {
        return Err(crate::Error::InvalidOperation(
            "the rendezvous names no fence".into(),
        ));
    }
    let mut f: Option<ID3D11Fence> = None;
    // SAFETY: `d5` is live, the handle was published by a copy of this crate
    // in this process and is kept open for the process lifetime, and `f` is a
    // valid out-parameter.
    com::hr("ID3D11Device5::OpenSharedFence", unsafe {
        d5.OpenSharedFence(HANDLE(view.fence_handle as *mut c_void), &mut f)
    })?;
    let fence = f.ok_or_else(|| {
        crate::Error::IoError(std::io::Error::other("OpenSharedFence returned no fence"))
    })?;
    Ok((fence, duplicate_raw_handle(view.fence_handle as RawHandle)?))
}

/// The device's adapter LUID, whether that adapter is WARP, and the adapter
/// itself, kept so `adapter_ptr` can hand it to CUDA. A device whose DXGI
/// chain does not answer reports a zero LUID and no adapter rather than
/// failing creation.
fn adapter_identity(dev: &ID3D11Device) -> ((i32, u32), bool, Option<IDXGIAdapter>) {
    let Ok(dxgi) = dev.cast::<IDXGIDevice>() else {
        return ((0, 0), false, None);
    };
    // SAFETY: `dxgi` is live.
    let adapter = match unsafe { dxgi.GetAdapter() } {
        Ok(a) => a,
        Err(e) => {
            log::debug!("IDXGIDevice::GetAdapter failed: {e}");
            return ((0, 0), false, None);
        }
    };
    // SAFETY: `adapter` is live.
    let desc = match unsafe { adapter.GetDesc() } {
        Ok(d) => d,
        Err(e) => {
            log::debug!("IDXGIAdapter::GetDesc failed: {e}");
            return ((0, 0), false, Some(adapter));
        }
    };
    let len = desc
        .Description
        .iter()
        .position(|&c| c == 0)
        .unwrap_or(desc.Description.len());
    let description = String::from_utf16_lossy(&desc.Description[..len]);
    let luid = (desc.AdapterLuid.HighPart, desc.AdapterLuid.LowPart);
    (luid, description.contains(WARP_DESCRIPTION), Some(adapter))
}

fn rendezvous_name() -> Vec<u16> {
    // SAFETY: no preconditions.
    let pid = unsafe { GetCurrentProcessId() };
    com::wide(&format!("Local\\edgefirst-d3d11-device-{pid}"))
}

/// A rendezvous mapping opened and left mapped for the process lifetime, so
/// `counter` stays valid for as long as any `D3d11Device` can signal.
struct RendezvousView {
    device_ptr: usize,
    fence_handle: usize,
    counter: &'static AtomicU64,
}

/// What [`publish_rendezvous`] left in the mapping.
enum Published {
    /// This copy published; the shared counter in the view it keeps mapped.
    Ours(&'static AtomicU64),
    /// Another copy published a live device first; what it published.
    Other(RendezvousView),
    /// The rendezvous could not be created or mapped. The device still works;
    /// other copies of this crate will just not find it.
    Unavailable,
}

/// The device pointer another copy of this crate published, after checking
/// it still answers QueryInterface. Stale or foreign mappings read as `None`.
///
/// This is the read-only peek; it unmaps before returning. Use
/// [`open_rendezvous`] to take the fence and counter with it.
///
/// The creation path needs the fence and counter too, so it opens; this
/// answers "what is published" for callers that only want the device.
pub(crate) fn read_rendezvous() -> Option<usize> {
    let name = rendezvous_name();
    // SAFETY: documented call with a NUL-terminated name; an error means no
    // copy of this crate has published in this process.
    let map = unsafe { OpenFileMappingW(FILE_MAP_READ.0, false, PCWSTR(name.as_ptr())) }.ok()?;
    // A mapping shorter than `RENDEZVOUS_BYTES` -- an older layout, or another
    // program on the same name -- maps to a null view and so reads as "no
    // rendezvous" rather than as bytes to trust.
    // SAFETY: `map` is a live mapping; a view shorter than the request fails.
    let view = unsafe { MapViewOfFile(map, FILE_MAP_READ, 0, 0, RENDEZVOUS_BYTES) };
    let result = if view.Value.is_null() {
        None
    } else {
        // SAFETY: the view is `RENDEZVOUS_BYTES` readable bytes laid out by
        // `publish_rendezvous`, page-aligned, and live until the unmap below.
        let version = unsafe { settled_version(&*(view.Value as *const AtomicU64)) };
        // Nothing else is read until the version says a record is there. A
        // version still at 0 or at the claiming marker means the holder is
        // mid-`write_unaligned` of exactly the word below, and reading it
        // would race that write; the acquire load above is what makes the
        // published record visible, as `published_record` relies on too.
        let ptr = if version == RENDEZVOUS_VERSION {
            // Only the pointer: reading the whole record would overlap the
            // counter word at `COUNTER_OFFSET`, which other copies mutate
            // atomically.
            // SAFETY: as above; the pointer word follows the version word.
            unsafe { std::ptr::read_unaligned(view.Value.byte_add(8) as *const u64) as usize }
        } else {
            0
        };
        // SAFETY: `view` is what `MapViewOfFile` just returned.
        let _ = unsafe { UnmapViewOfFile(view) };
        (ptr != 0 && device_pointer_is_live(ptr)).then_some(ptr)
    };
    // SAFETY: `map` is the handle `OpenFileMappingW` returned.
    let _ = unsafe { CloseHandle(map) };
    result
}

/// Reads the version word until it holds a published version, waiting out the
/// two transient values a publisher passes through.
///
/// `CreateFileMappingW` zero-fills a new section and the record is written
/// several stores later, so `0` on a section that exists means "a copy is
/// about to claim this" and [`RENDEZVOUS_CLAIMING`] means "a copy is filling
/// it in now". A reader that treated either as stale would conclude no device
/// is published and mint a second one, which is exactly what the rendezvous
/// exists to prevent. Returns the word as it stands after
/// [`RENDEZVOUS_CLAIM_SPINS`] yields when it never settles.
fn settled_version(version: &AtomicU64) -> u64 {
    for turn in 0..RENDEZVOUS_CLAIM_SPINS {
        let seen = version.load(Ordering::Acquire);
        if seen != 0 && seen != RENDEZVOUS_CLAIMING {
            return seen;
        }
        claim_backoff(turn);
    }
    version.load(Ordering::Acquire)
}

/// One turn of the claim wait: yields while the holder is likely still
/// running, sleeps after that so a preempted holder gets its core back.
fn claim_backoff(turn: u32) {
    if turn < RENDEZVOUS_CLAIM_YIELDS {
        std::thread::yield_now();
    } else {
        std::thread::sleep(std::time::Duration::from_micros(RENDEZVOUS_CLAIM_SLEEP_US));
    }
}

/// Opens the rendezvous for writing and keeps the view mapped, so the shared
/// counter can be handed out as `&'static`. Returns `None` when nothing is
/// published, the layout does not match, or the device named is gone.
fn open_rendezvous() -> Option<RendezvousView> {
    let name = rendezvous_name();
    // SAFETY: documented call with a NUL-terminated name; an error means no
    // copy of this crate has published in this process.
    let map = unsafe { OpenFileMappingW(FILE_MAP_WRITE.0, false, PCWSTR(name.as_ptr())) }.ok()?;
    // A mapping shorter than `RENDEZVOUS_BYTES` maps to a null view, so an
    // older layout on the same name reads as "no rendezvous".
    // SAFETY: `map` is a live mapping; a view shorter than the request fails.
    let view = unsafe { MapViewOfFile(map, FILE_MAP_WRITE, 0, 0, RENDEZVOUS_BYTES) };
    if view.Value.is_null() {
        // SAFETY: `map` is the handle `OpenFileMappingW` returned.
        let _ = unsafe { CloseHandle(map) };
        return None;
    }
    // SAFETY: the view is `RENDEZVOUS_BYTES` readable bytes laid out by
    // `publish_rendezvous`, page-aligned, and live until the unmap below.
    match unsafe { published_record(view) } {
        Some(record) => Some(record),
        None => {
            // SAFETY: `view` and `map` are what the two calls above returned.
            unsafe {
                let _ = UnmapViewOfFile(view);
                let _ = CloseHandle(map);
            }
            None
        }
    }
}

/// Reads a published record out of a mapped rendezvous view, or `None` when
/// the version does not match this build, no claim ever completed, or the
/// device named has gone.
///
/// The view is kept mapped by the caller on the `Some` path: `counter` points
/// into it, and every copy of this crate on this device allocates fence values
/// from that one word.
///
/// # Safety
///
/// `view` must be a live mapping of at least `RENDEZVOUS_BYTES` page-aligned
/// bytes that outlives the process on the `Some` path.
unsafe fn published_record(view: MEMORY_MAPPED_VIEW_ADDRESS) -> Option<RendezvousView> {
    // SAFETY: the caller guarantees a page-aligned view at least
    // `RENDEZVOUS_BYTES` long, so both `u64` words are aligned.
    let version = unsafe { settled_version(&*(view.Value as *const AtomicU64)) };
    if version != RENDEZVOUS_VERSION {
        return None;
    }
    // Read after the acquire load above, which pairs with the publisher's
    // release store of the version word, so both are the values it wrote.
    // Only these two: the counter word at `COUNTER_OFFSET` is mutated
    // atomically by other copies and is taken by reference instead.
    // SAFETY: as above; both words are inside the view.
    let (device_ptr, fence_handle) = unsafe {
        (
            std::ptr::read_unaligned(view.Value.byte_add(8) as *const u64) as usize,
            std::ptr::read_unaligned(view.Value.byte_add(16) as *const u64) as usize,
        )
    };
    if device_ptr == 0 || !device_pointer_is_live(device_ptr) {
        return None;
    }
    // SAFETY: the view is page-aligned, so the `u64` at `COUNTER_OFFSET` is
    // aligned, and the caller keeps the view for the process lifetime.
    let counter = unsafe { &*(view.Value.byte_add(COUNTER_OFFSET) as *const AtomicU64) };
    Some(RendezvousView {
        device_ptr,
        fence_handle,
        counter,
    })
}

/// Publishes `device_ptr` and `fence_handle` under this process's rendezvous
/// name and returns the shared counter.
///
/// `CreateFileMappingW` succeeds on an existing name and reports
/// `ERROR_ALREADY_EXISTS`, but that flag is not what decides the winner: the
/// section is zero-filled at creation and the record is written several stores
/// later, so two copies can both see "already exists" or both see "created".
/// The version word is the claim instead -- one interlocked move off `0` (or
/// off a version naming a device that has gone), the record written by the
/// winner alone, and the version release-stored last so a reader either sees
/// the whole record or waits for it. When another copy holds a live device,
/// that device is returned and its record is left alone, counter word
/// included: zeroing a counter another copy has already allocated from would
/// make its next `signal` non-increasing, which `ID3D11DeviceContext4::Signal`
/// rejects.
///
/// A rendezvous that cannot be created, mapped or claimed is a warning, never
/// a failure: the device this copy holds still works.
fn publish_rendezvous(device_ptr: usize, fence_handle: usize) -> Published {
    let name = rendezvous_name();
    // SAFETY: documented call; an anonymous, page-file-backed mapping named
    // for this process.
    let map = unsafe {
        CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            None,
            PAGE_READWRITE,
            0,
            RENDEZVOUS_BYTES as u32,
            PCWSTR(name.as_ptr()),
        )
    };
    let map = match map {
        Ok(map) => map,
        Err(_) => {
            log::warn!(
                "cannot create the D3D11 rendezvous mapping ({}); other copies of this crate in this process will create their own device",
                std::io::Error::last_os_error()
            );
            return Published::Unavailable;
        }
    };
    // SAFETY: `map` is live and `RENDEZVOUS_BYTES` long.
    let view = unsafe { MapViewOfFile(map, FILE_MAP_WRITE, 0, 0, RENDEZVOUS_BYTES) };
    if view.Value.is_null() {
        log::warn!(
            "cannot map the D3D11 rendezvous ({}); other copies of this crate in this process will create their own device",
            std::io::Error::last_os_error()
        );
        // SAFETY: `map` is the handle `CreateFileMappingW` returned.
        let _ = unsafe { CloseHandle(map) };
        return Published::Unavailable;
    }
    // SAFETY: the view is `RENDEZVOUS_BYTES` writable page-aligned bytes, so
    // the `u64` at offset 0 is aligned, and it is live until the unmap below.
    let version = unsafe { &*(view.Value as *const AtomicU64) };
    for turn in 0..RENDEZVOUS_CLAIM_SPINS {
        if version
            .compare_exchange(0, RENDEZVOUS_CLAIMING, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            // SAFETY: this copy holds the claim, and the view is live and long
            // enough for the whole record.
            return unsafe { write_claimed_record(view, device_ptr, fence_handle) };
        }
        match version.load(Ordering::Acquire) {
            // Another copy is between its claim and its release store, or the
            // section exists and is about to be claimed.
            0 | RENDEZVOUS_CLAIMING => claim_backoff(turn),
            seen if seen == RENDEZVOUS_VERSION => {
                // Any live entry is adopted, including one that names the
                // device this copy is publishing: two copies adopting the same
                // host device race here, and overwriting would replace a live
                // fence handle and counter with this copy's own.
                // SAFETY: as above; the view is kept mapped on the `Some` path.
                if let Some(record) = unsafe { published_record(view) } {
                    return Published::Other(record);
                }
                // The record names a device that has gone. Take the claim back
                // off it; if another copy got there first the loop re-reads.
                if version
                    .compare_exchange(
                        seen,
                        RENDEZVOUS_CLAIMING,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    // SAFETY: as on the first claim above.
                    return unsafe { write_claimed_record(view, device_ptr, fence_handle) };
                }
            }
            other => {
                log::warn!(
                    "the D3D11 rendezvous for this process holds version {other:#x}, not \
                     {RENDEZVOUS_VERSION}; this copy keeps its own device and other copies \
                     of this crate will create theirs"
                );
                return unmapped(view, map);
            }
        }
    }
    log::warn!(
        "the D3D11 rendezvous claim for this process never settled; this copy keeps its own device"
    );
    unmapped(view, map)
}

/// Releases a rendezvous view and its section handle and reports that no
/// rendezvous is usable.
fn unmapped(view: MEMORY_MAPPED_VIEW_ADDRESS, map: HANDLE) -> Published {
    // SAFETY: `view` and `map` are what `MapViewOfFile` and
    // `CreateFileMappingW` returned and nothing else holds either.
    unsafe {
        let _ = UnmapViewOfFile(view);
        let _ = CloseHandle(map);
    }
    Published::Unavailable
}

/// Fills in a rendezvous record this copy has claimed and publishes it with a
/// release store of the version word.
///
/// The counter word is left exactly as it stands: a section this copy created
/// is zero-filled, and one taken over from a departed device may already have
/// handed out values, so zeroing it would make the next `signal` allocate a
/// value the fence has already passed --
/// `ID3D11DeviceContext4::Signal` rejects a non-increasing value, and
/// `event_for` on one returns an already-set event, a false completion.
///
/// The view and the mapping handle stay open for the process lifetime: the
/// handle is what keeps the name resolvable for copies that initialize later,
/// and the view is what the returned counter points into. `HANDLE` has no
/// destructor, so leaving both alone is all that takes. The device named stays
/// alive because `DEVICE` is a static that is never dropped.
///
/// # Safety
///
/// The caller must hold the claim on `view`'s version word, and `view` must be
/// a live page-aligned mapping of at least `RENDEZVOUS_BYTES` bytes.
unsafe fn write_claimed_record(
    view: MEMORY_MAPPED_VIEW_ADDRESS,
    device_ptr: usize,
    fence_handle: usize,
) -> Published {
    // SAFETY: the caller guarantees the view is long enough, and the claim
    // makes this copy the only writer of these two words.
    unsafe {
        std::ptr::write_unaligned(view.Value.byte_add(8) as *mut u64, device_ptr as u64);
        std::ptr::write_unaligned(view.Value.byte_add(16) as *mut u64, fence_handle as u64);
    }
    // SAFETY: the view is page-aligned, so both `u64` words are aligned, and
    // it outlives the process.
    let (version, counter) = unsafe {
        (
            &*(view.Value as *const AtomicU64),
            &*(view.Value.byte_add(COUNTER_OFFSET) as *const AtomicU64),
        )
    };
    // Stored last, so a reader that sees this version sees the two writes
    // above with it.
    version.store(RENDEZVOUS_VERSION, Ordering::Release);
    Published::Ours(counter)
}

fn device_pointer_is_live(ptr: usize) -> bool {
    // SAFETY: the pointer was published by a copy of this crate in this
    // process. The version field and this QueryInterface reduce, but cannot
    // eliminate, the risk of reading one whose publisher has gone.
    unsafe { IUnknown::from_raw_borrowed(&(ptr as *mut c_void)) }
        .is_some_and(|u| u.cast::<ID3D11Device>().is_ok())
}

#[cfg_attr(not(feature = "static"), allow(dead_code))] // see the struct's own note
impl D3d11Device {
    /// The `ID3D11Device*` itself, borrowed: no reference is transferred, so
    /// a caller that keeps it must AddRef.
    pub fn raw(&self) -> *mut c_void {
        self.dev.as_raw()
    }

    pub(crate) fn dev(&self) -> &ID3D11Device {
        &self.dev
    }

    /// Opens shared NT handles with `OpenSharedResource1`.
    pub(crate) fn dev1(&self) -> Option<&ID3D11Device1> {
        self.dev1.as_ref()
    }

    /// Opens a producer's shared fence with `OpenSharedFence`.
    pub(crate) fn dev5(&self) -> Option<&ID3D11Device5> {
        self.dev5.as_ref()
    }

    pub(crate) fn ctx(&self) -> &ID3D11DeviceContext {
        &self.ctx
    }

    /// The context's `ID3D11DeviceContext4`, for the GPU-side `Wait` a
    /// texture opened from a shared handle issues on the producer's fence.
    pub(crate) fn ctx4(&self) -> Option<&ID3D11DeviceContext4> {
        self.ctx4.as_ref()
    }

    /// This copy's fence, the one `signal` advances and `last_gpu_write`
    /// values name. For a GPU-side `ID3D11DeviceContext4::Wait` on a
    /// locally recorded write, which a CPU map queues before its staging
    /// refresh. `None` on a device with no fence at all.
    pub(crate) fn fence(&self) -> Option<&ID3D11Fence> {
        self.fence.as_ref()
    }

    pub fn is_warp(&self) -> bool {
        self.is_warp
    }

    pub fn adapter_label(&self) -> &str {
        &self.adapter_label
    }

    /// The adapter LUID as `(high, low)`.
    pub fn luid(&self) -> (i32, u32) {
        self.luid
    }

    pub fn creation_flags(&self) -> u32 {
        self.creation_flags
    }

    pub fn signal_supported(&self) -> bool {
        self.fence.is_some() && self.ctx4.is_some()
    }

    /// Queues a fence signal after everything issued so far on the immediate
    /// context, submits the context, and returns the value. Call it on the
    /// thread that just issued the GPU work you want the value to cover.
    ///
    /// The submission is part of the contract, not an optimisation: a consumer
    /// in another process waits on this fence with nothing of ours queued
    /// behind it, so an unsubmitted Signal would leave it waiting on a value
    /// that never arrives.
    pub fn signal(&self) -> Option<u64> {
        self.signal_with(true)
    }

    /// [`signal`](Self::signal) without the submission: the value is
    /// allocated and its `Signal` queued, but the immediate context is left
    /// for the caller to submit with [`flush`](Self::flush).
    ///
    /// For a producer issuing several pieces of work that end in one
    /// submission -- the image crate's deferred convert batch -- where one
    /// `Flush` per item would undo the batching the caller asked for. The
    /// values are still allocated in order under the device lock, so they
    /// order the items among themselves; what they do not carry until the
    /// flush is the promise a `signal` makes, that a waiter in another
    /// process sees the value arrive.
    pub fn signal_deferred(&self) -> Option<u64> {
        self.signal_with(false)
    }

    /// Submits the immediate context. Pairs with
    /// [`signal_deferred`](Self::signal_deferred): every value queued since
    /// the last submission becomes visible to waiters here.
    pub fn flush(&self) {
        // SAFETY: the immediate context is live for the process's lifetime.
        unsafe { self.ctx.Flush() };
    }

    fn signal_with(&self, submit: bool) -> Option<u64> {
        let (fence, ctx4) = (self.fence.as_ref()?, self.ctx4.as_ref()?);
        // Allocating the value and issuing its Signal must be one step: two
        // threads that allocate 1 and 2 and then issue Signal(2) first drive
        // the fence timeline backwards. `ID3D11Multithread`'s critical section
        // is the device's own -- one object shared by every copy of this crate
        // on the device -- so it orders signals across copies as well as
        // threads. A device without the interface gets a process-local lock,
        // which is all this copy can do on its own.
        let _local = self
            .mt
            .is_none()
            .then(|| self.signal_lock.lock().unwrap_or_else(|e| e.into_inner()));
        if let Some(mt) = &self.mt {
            // SAFETY: `mt` is live and every path below leaves the section.
            unsafe { mt.Enter() };
        }
        let value = self.counter.word().fetch_add(1, Ordering::AcqRel) + 1;
        // SAFETY: both interfaces are live; the value is fresh.
        let signalled = unsafe { ctx4.Signal(fence, value) };
        // Submitted inside the same bracket as the Signal it flushes. A queued
        // Signal is not visible to anyone until the context is submitted, so a
        // waiter in another process -- which has no unrelated work of ours to
        // ride along with -- would block on a value this device has recorded
        // but never sent. Flushing here is what makes the value the caller
        // returns mean "queued and submitted"; `signal_deferred` moves that
        // promise to the caller's own `flush`.
        if submit && signalled.is_ok() {
            // SAFETY: the immediate context is live for the process's lifetime.
            unsafe { self.ctx().Flush() };
        }
        if let Some(mt) = &self.mt {
            // SAFETY: `mt` is live and this thread entered above.
            unsafe { mt.Leave() };
        }
        if let Err(e) = signalled {
            log::warn!("ID3D11DeviceContext4::Signal({value}) failed: {e}");
            return None;
        }
        Some(value)
    }

    /// The newest value allocated on the shared counter. Its `Signal` was
    /// issued under the device lock and succeeded, unless `signal` returned
    /// `None` for that value: a `Signal` the immediate context rejects is not
    /// recoverable, so the counter keeps the value and the fence never
    /// reaches it.
    pub fn last_signalled(&self) -> u64 {
        self.counter.word().load(Ordering::Acquire)
    }

    pub fn completed_value(&self) -> u64 {
        // SAFETY: `fence` is live.
        self.fence
            .as_ref()
            .map_or(0, |f| unsafe { f.GetCompletedValue() })
    }

    /// A manual-reset event set when the fence reaches `value`; set already
    /// when the value is complete.
    pub fn event_for(&self, value: u64) -> std::io::Result<OwnedHandle> {
        let fence = self
            .fence
            .as_ref()
            .ok_or_else(|| std::io::Error::other("no fence"))?;
        event_on(fence, value)
    }

    /// A duplicate of the fence's shared NT handle for GPU waiters.
    pub fn fence_shared_handle(&self) -> std::io::Result<OwnedHandle> {
        let src = self
            .fence_handle
            .as_ref()
            .ok_or_else(|| std::io::Error::other("no fence"))?;
        duplicate_handle(src)
    }

    /// The fence's own NT handle value, whichever fence this copy holds. For
    /// the rendezvous, which publishes it precisely so the other copies can
    /// open it. Zero without a fence.
    pub(crate) fn fence_handle_value(&self) -> usize {
        self.fence_handle
            .as_ref()
            .map_or(0, |h| h.as_raw_handle() as usize)
    }

    /// Whether this copy signals the fence the process agreed on, rather than
    /// one of its own.
    ///
    /// False in the degraded case `fence_and_counter` warns about: a copy that
    /// adopted a published device but could not open its fence, or one with no
    /// rendezvous at all. Its values come from a counter no other copy reads.
    pub(crate) fn fence_is_shared(&self) -> bool {
        matches!(self.counter, Counter::Shared(_))
    }

    /// The fence handle value to hand to a consumer outside this copy, or `0`.
    ///
    /// Zero whenever the fence is not the shared one: a completion *value*
    /// recorded by another copy paired with this copy's private fence names a
    /// point that fence never reaches, so a consumer's
    /// `ID3D11DeviceContext4::Wait` would block its context forever -- or,
    /// past the value, return at once and read a texture still being written.
    /// A zero handle is what `from_parts` and `reference_handle_bytes` read as
    /// "nothing to wait on", which is the honest answer here.
    pub(crate) fn exported_fence_handle_value(&self) -> usize {
        if self.fence_is_shared() {
            self.fence_handle_value()
        } else {
            0
        }
    }

    /// The `IDXGIAdapter*` of this device, borrowed, for `cudaD3D11GetDevice`.
    #[allow(dead_code)]
    pub(crate) fn adapter_ptr(&self) -> *mut c_void {
        self.adapter
            .as_ref()
            .map_or(std::ptr::null_mut(), |a| a.as_raw())
    }
}

/// A manual-reset event set when `fence` reaches `value`; set already when
/// the value is complete.
fn event_on(fence: &ID3D11Fence, value: u64) -> std::io::Result<OwnedHandle> {
    // SAFETY: documented call; manual reset, initially unsignalled, unnamed.
    let ev = unsafe { CreateEventW(None, true, false, PCWSTR::null()) }
        .map_err(|_| std::io::Error::last_os_error())?;
    // SAFETY: `fence` is live and `ev` is the event just created.
    if let Err(e) = unsafe { fence.SetEventOnCompletion(value, ev) } {
        // SAFETY: `ev` is the handle `CreateEventW` returned and nothing else holds it.
        let _ = unsafe { CloseHandle(ev) };
        return Err(std::io::Error::other(format!(
            "SetEventOnCompletion: {:#010x}",
            e.code().0 as u32
        )));
    }
    // SAFETY: this call owns `ev`.
    Ok(unsafe { OwnedHandle::from_raw_handle(ev.0) })
}

/// Blocks the calling thread until `fence` reaches `value`, or `timeout_ms`
/// elapses.
///
/// The CPU-side stand-in for the GPU-side `ID3D11DeviceContext4::Wait` a
/// device without that interface cannot issue: an importer that cannot order
/// the producer's work behind its own has to be sure the work is done before
/// it hands the tensor back. Bounded rather than infinite so a value the
/// producer never signals costs a warning instead of a hung thread.
#[cfg_attr(not(feature = "static"), allow(dead_code))] // only the texture storage waits
pub(crate) fn wait_cpu_for(
    fence: &ID3D11Fence,
    value: u64,
    timeout_ms: u32,
) -> std::io::Result<()> {
    let ev = event_on(fence, value)?;
    // SAFETY: `ev` is a live event handle owned by this scope.
    match unsafe { WaitForSingleObject(HANDLE(ev.as_raw_handle()), timeout_ms) } {
        WAIT_OBJECT_0 => Ok(()),
        other => Err(std::io::Error::other(format!(
            "WaitForSingleObject on fence value {value}: {:#010x}",
            other.0
        ))),
    }
}

fn duplicate(src: HANDLE) -> std::io::Result<OwnedHandle> {
    let mut out = HANDLE(std::ptr::null_mut());
    // SAFETY: documented call; both process arguments are the current-process
    // pseudo-handle, `src` is live for the call and `out` is a valid local.
    unsafe {
        DuplicateHandle(
            GetCurrentProcess(),
            src,
            GetCurrentProcess(),
            &mut out,
            0,
            false,
            DUPLICATE_SAME_ACCESS,
        )
    }
    .map_err(|_| std::io::Error::last_os_error())?;
    // SAFETY: this call owns `out`.
    Ok(unsafe { OwnedHandle::from_raw_handle(out.0) })
}

pub(crate) fn duplicate_handle(src: &OwnedHandle) -> std::io::Result<OwnedHandle> {
    duplicate(HANDLE(src.as_raw_handle()))
}

/// `duplicate_handle` over a handle this crate does not own. Ownership of
/// `raw` stays with the caller.
pub(crate) fn duplicate_raw_handle(raw: RawHandle) -> std::io::Result<OwnedHandle> {
    duplicate(HANDLE(raw))
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Foundation::WAIT_OBJECT_0;
    use windows::Win32::System::Threading::WaitForSingleObject;

    #[test]
    fn device_is_a_process_singleton_with_protection_and_a_fence() {
        let a = device().expect("D3D11 device");
        let b = device().unwrap();
        assert!(std::ptr::eq(a, b));
        assert_ne!(a.raw(), std::ptr::null_mut());
        assert!(
            a.signal_supported(),
            "ID3D11Device5 is Windows 10 1703+; this box has it"
        );
        assert!(
            a.creation_flags() & D3D11_CREATE_DEVICE_BGRA_SUPPORT.0 != 0,
            "BGRA_SUPPORT requested"
        );
    }

    #[test]
    fn signal_then_event_is_set_and_completed_value_catches_up() {
        let d = device().unwrap();
        let v = d.signal().expect("fence");
        let ev = d.event_for(v).unwrap();
        // SAFETY: `ev` is a live event handle owned by this scope.
        let w = unsafe { WaitForSingleObject(HANDLE(ev.as_raw_handle()), 5000) };
        assert_eq!(w, WAIT_OBJECT_0);
        assert!(d.completed_value() >= v);
        // The shared counter, which the sibling tests in this binary also
        // allocate from, so the newest value is at least this one.
        assert!(d.last_signalled() >= v);
    }

    #[test]
    fn use_external_device_after_first_use_is_refused() {
        let d = device().unwrap();
        // SAFETY: `d.raw()` is the live process device, which `DEVICE` holds
        // for the process lifetime.
        let err = unsafe { use_external_device(d.raw()) }.unwrap_err();
        assert!(matches!(err, crate::Error::InvalidOperation(_)));
    }

    /// The second half of that refusal, the one a *sibling* copy of this
    /// crate hits: its own `DEVICE` is empty, but the device this process
    /// already has is published in the rendezvous. This binary is a single
    /// copy, so `DEVICE` short-circuits before the rendezvous check is
    /// reached -- what this pins is that the state the check reads is
    /// genuinely there once the device exists, which is the whole premise of
    /// the branch.
    #[test]
    fn use_external_device_sees_the_device_a_sibling_copy_published() {
        let d = device().unwrap();
        assert_eq!(
            read_rendezvous(),
            Some(d.raw() as usize),
            "the device is published, so a sibling copy's rendezvous check finds it"
        );
        // SAFETY: `d.raw()` is the live process device, which `DEVICE` holds
        // for the process lifetime.
        let err = unsafe { use_external_device(d.raw()) }.unwrap_err();
        assert!(matches!(err, crate::Error::InvalidOperation(_)));
    }

    #[test]
    fn rendezvous_mapping_names_this_process_and_carries_the_device() {
        let d = device().unwrap();
        let published = read_rendezvous().expect("mapping exists after device()");
        assert_eq!(published, d.raw() as usize);
    }

    /// What a second, independently linked copy of this crate does: adopt the
    /// published device and get the published fence and counter with it.
    #[test]
    fn adopted_device_shares_the_fence_and_counter() {
        let first = device().unwrap();
        let before = first.last_signalled();
        let view = open_rendezvous().expect("the process published a rendezvous");
        let second = adopt_published(view).expect("adopt the published device");
        assert_eq!(second.raw(), first.raw());
        assert_ne!(
            second.fence_handle_value(),
            0,
            "the adopted copy duplicated the published handle"
        );

        let value = second.signal().expect("fence");
        assert!(
            value > before,
            "the second copy allocated {value} from the shared counter, not past {before}"
        );
        assert_eq!(
            first.last_signalled(),
            second.last_signalled(),
            "both copies read one counter"
        );

        let ev = first.event_for(value).unwrap();
        // SAFETY: `ev` is a live event handle owned by this scope.
        let w = unsafe { WaitForSingleObject(HANDLE(ev.as_raw_handle()), 5000) };
        assert_eq!(
            w, WAIT_OBJECT_0,
            "the first copy's fence is the one the second signalled"
        );
        assert!(first.completed_value() >= value);
    }

    /// The claim window the atomic publication closes: a reader that saw the
    /// marker (or the zero-filled word a fresh section starts at) and called
    /// the record stale would publish a device of its own, and the process
    /// would run two.
    #[test]
    fn a_reader_waits_out_a_claim_instead_of_calling_the_record_stale() {
        let word = std::sync::Arc::new(AtomicU64::new(RENDEZVOUS_CLAIMING));
        let claimer = std::sync::Arc::clone(&word);
        let publisher = std::thread::spawn(move || {
            std::thread::yield_now();
            claimer.store(RENDEZVOUS_VERSION, Ordering::Release);
        });
        assert_eq!(settled_version(&word), RENDEZVOUS_VERSION);
        publisher.join().unwrap();
        // A word that never settles is reported as it stands rather than spun
        // on forever.
        assert_eq!(settled_version(&AtomicU64::new(0)), 0);
    }

    /// Two copies publishing at once converge on one device. Both find the
    /// device this process already published, which is the outcome the claim
    /// guarantees whichever order they arrive in.
    #[test]
    fn two_threads_racing_to_publish_end_on_one_device() {
        let d = device().unwrap();
        let (ptr, fence_handle) = (d.raw() as usize, d.fence_handle_value());
        let seen = std::thread::scope(|s| {
            let threads: Vec<_> = (0..2)
                .map(|_| {
                    s.spawn(move || match publish_rendezvous(ptr, fence_handle) {
                        Published::Other(view) => Some(view.device_ptr),
                        Published::Ours(_) => Some(ptr),
                        Published::Unavailable => None,
                    })
                })
                .collect();
            threads
                .into_iter()
                .map(|t| t.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(seen, vec![Some(ptr), Some(ptr)]);
    }

    /// The degraded state `fence_and_counter` falls back to: a copy that could
    /// not open the published fence signals one of its own, whose values no
    /// other copy can compare with. It exports no fence handle, so a consumer
    /// reads "no completion" rather than a value paired with the wrong fence.
    ///
    /// This is the device half. The transport half -- that a zero handle takes
    /// the recorded *value* with it, since `blob::import` refuses a value with
    /// no fence beside it -- is `blob::tests::
    /// a_reference_plane_drops_the_value_with_the_fence`, tested against the
    /// rule rather than through `export`: the export path reads the process
    /// device (`crate::d3d11::device()`), which a locally wrapped copy cannot
    /// stand in for.
    #[test]
    fn a_copy_with_a_private_fence_exports_no_fence_handle() {
        let d = device().unwrap();
        assert!(d.fence_is_shared());
        assert_eq!(d.exported_fence_handle_value(), d.fence_handle_value());
        assert_ne!(d.exported_fence_handle_value(), 0);

        // `wrap` with no published record is exactly what a copy gets when the
        // published fence cannot be opened: its own fence, its own counter.
        let private = wrap(d.dev().clone(), "private fence".into(), None, None).unwrap();
        assert!(!private.fence_is_shared());
        assert_ne!(
            private.fence_handle_value(),
            0,
            "it has a fence, just not the one the process agreed on"
        );
        assert_eq!(
            private.exported_fence_handle_value(),
            0,
            "a descriptor or a blob from this copy carries no completion at all"
        );
        // And the value it would have paired with is real, which is exactly
        // why the transport has to drop it too.
        assert!(private.signal().is_some_and(|v| v > 0));
    }

    #[test]
    fn duplicate_handle_round_trips_the_fence_handle() {
        let d = device().unwrap();
        let (a, b) = (
            d.fence_shared_handle().unwrap(),
            d.fence_shared_handle().unwrap(),
        );
        assert_ne!(a.as_raw_handle(), b.as_raw_handle());
        assert!(!a.as_raw_handle().is_null() && !b.as_raw_handle().is_null());
        drop(a);
        let mut fence: Option<ID3D11Fence> = None;
        // SAFETY: `b` is a live shared-fence handle and `fence` is a valid out-parameter.
        unsafe {
            d.dev5()
                .unwrap()
                .OpenSharedFence(HANDLE(b.as_raw_handle()), &mut fence)
        }
        .expect("the surviving duplicate still names the fence");
        assert!(fence.is_some());
    }
}
