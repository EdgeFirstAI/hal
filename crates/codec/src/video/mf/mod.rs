// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! H.264 through the Windows Media Foundation transforms (MFTs).
//!
//! The encoder prefers a hardware MFT (NVENC, Quick Sync, AMF) and falls back
//! to Microsoft's software encoder; the decoder uses Microsoft's H.264
//! decoder. Both run on the caller's thread: an asynchronous (hardware)
//! transform is driven by polling its event queue rather than through
//! callbacks, so no D3D11 work ever runs on a Media Foundation thread.
//!
//! Pixels stay on the GPU where the transform takes a DXGI device manager,
//! which is given HAL's own D3D11 device. An encoder reads a native NV12
//! tensor's texture directly; a decoder writes into its own texture array,
//! from which each picture is copied, GPU to GPU, into a native NV12 tensor
//! of the pool. A transform without D3D11 support (Microsoft's software
//! encoder, a decoder on an adapter without video decoding) moves pixels
//! through system memory instead.
//!
//! The flat Media Foundation functions are resolved from `mfplat.dll` at
//! run time, so the crate loads on a system without Media Foundation
//! (Windows N without the Media Feature Pack), where the codecs report
//! [`CodecError::NoDevice`].

pub(crate) mod copy;
pub(crate) mod decoder;
pub(crate) mod encoder;
mod transform;

use std::ffi::c_void;
use std::sync::OnceLock;

use edgefirst_tensor::PixelFormat;
use libloading::os::windows::{Library, LOAD_LIBRARY_SEARCH_SYSTEM32};
use windows::core::{Interface, GUID, HRESULT, PWSTR};
use windows::Win32::Graphics::Direct3D11::{ID3D11Device, ID3D11Texture2D};
use windows::Win32::Media::MediaFoundation::{
    IMFActivate, IMFAttributes, IMFDXGIBuffer, IMFDXGIDeviceManager, IMFMediaBuffer, IMFMediaType,
    IMFSample, MFMediaType_Video, MFT_FRIENDLY_NAME_Attribute, MFVideoFormat_H264,
    MFVideoFormat_NV12, MFVideoFormat_YUY2, MFSTARTUP_LITE, MFT_CATEGORY_VIDEO_DECODER,
    MFT_CATEGORY_VIDEO_ENCODER, MFT_ENUM_FLAG_ASYNCMFT, MFT_ENUM_FLAG_HARDWARE,
    MFT_ENUM_FLAG_SORTANDFILTER, MFT_ENUM_FLAG_SYNCMFT, MFT_REGISTER_TYPE_INFO, MF_VERSION,
};
use windows::Win32::System::Com::{
    CoInitializeEx, CoTaskMemFree, CoUninitialize, COINIT_MULTITHREADED,
};

use crate::{Capability, Codec, CodecError, Direction, Engine, Result, Transfer};

pub(crate) use transform::{Output, Transform};

/// Adds the H.264 transforms present here to `out`, as the encoder and
/// decoder would find them. A transform that takes a device manager over
/// HAL's D3D11 device works on textures: an encoder reads native NV12
/// textures in place at best, and a decoder (DXVA) decodes on the GPU and
/// copies each picture there. Each such candidate is activated to ask it,
/// as opening it would.
pub(crate) fn capabilities(out: &mut Vec<Capability>) {
    let Ok(platform) = Platform::start() else {
        return;
    };
    let manager = platform.device_manager();
    let takes_textures = |c: Candidate| {
        manager
            .as_ref()
            .is_some_and(|m| Transform::activate(c).is_ok_and(|t| t.set_device_manager(m)))
    };
    let mut encoders: Vec<Capability> = Vec::new();
    for (subtype, format) in [
        (MFVideoFormat_NV12, PixelFormat::Nv12),
        (MFVideoFormat_YUY2, PixelFormat::Yuyv),
    ] {
        let candidates = platform
            .transforms(
                MFT_CATEGORY_VIDEO_ENCODER,
                Some(subtype),
                MFVideoFormat_H264,
                true,
                encoder::ENV_ENCODER,
            )
            .unwrap_or_default();
        for c in candidates {
            if let Some(e) = encoders.iter_mut().find(|e| e.device == c.name) {
                e.formats.push(format);
                continue;
            }
            let (name, hardware) = (c.name.clone(), c.hardware);
            // The encoder gives a device manager only for NV12 input.
            let textures = format == PixelFormat::Nv12 && takes_textures(c);
            encoders.push(Capability {
                engine: Engine::MediaFoundation,
                device: name,
                codec: Codec::H264,
                direction: Direction::Encode,
                hardware,
                formats: vec![format],
                transfer: if textures {
                    Transfer::ZeroCopy
                } else {
                    Transfer::CpuCopy
                },
            });
        }
    }
    out.extend(encoders);
    for codec in [Codec::H264, Codec::Mjpeg] {
        let subtype = decoder::input_subtype(codec).expect("a decoder codec");
        let decoders = platform
            .transforms(
                MFT_CATEGORY_VIDEO_DECODER,
                Some(subtype),
                MFVideoFormat_NV12,
                decoder::hardware_first(codec),
                decoder::ENV_DECODER,
            )
            .unwrap_or_default();
        for c in decoders {
            let (name, hardware) = (c.name.clone(), c.hardware);
            let dxva = takes_textures(c);
            out.push(Capability {
                engine: Engine::MediaFoundation,
                device: name,
                codec,
                direction: Direction::Decode,
                hardware: hardware || dxva,
                formats: vec![PixelFormat::Nv12],
                transfer: if dxva {
                    Transfer::GpuCopy
                } else {
                    Transfer::CpuCopy
                },
            });
        }
    }
}

/// HAL's process D3D11 device, which tensors' textures live on.
pub(crate) fn hal_device() -> Result<ID3D11Device> {
    let raw = edgefirst_tensor::d3d11::backend_device()
        .map_err(|e| CodecError::NoDevice(format!("no D3D11 device for Media Foundation: {e}")))?;
    // SAFETY: the tensor crate's device lives for the process; borrowing
    // takes no reference and `cloned` takes the one this keeps.
    unsafe { ID3D11Device::from_raw_borrowed(&raw) }
        .cloned()
        .ok_or_else(|| CodecError::NoDevice("the tensor crate reported a null D3D11 device".into()))
}

/// Whether `t` is a native two-plane NV12 texture, which Media Foundation
/// reads and writes on the GPU.
pub(crate) fn is_native_nv12(t: &edgefirst_tensor::TensorDyn) -> bool {
    t.d3d11_layout().is_some_and(|l| l.is_native_nv12())
}

/// The D3D11 texture behind a texture tensor.
pub(crate) fn tensor_texture(t: &edgefirst_tensor::TensorDyn) -> Result<ID3D11Texture2D> {
    let raw = t
        .d3d11_texture()
        .ok_or_else(|| CodecError::InvalidInput("the tensor is not a D3D11 texture".into()))?;
    // SAFETY: the tensor keeps its texture alive while it lives, which spans
    // every use of the returned reference; `cloned` takes its own.
    unsafe { ID3D11Texture2D::from_raw_borrowed(&raw) }
        .cloned()
        .ok_or_else(|| CodecError::InvalidInput("the tensor has a null texture".into()))
}

/// The texture behind a sample's first buffer and the subresource it names,
/// when the sample holds a D3D11 texture rather than system memory.
pub(crate) fn sample_texture(sample: &IMFSample) -> Option<(ID3D11Texture2D, u32)> {
    // SAFETY: a live sample.
    let buffer = unsafe { sample.GetBufferByIndex(0) }.ok()?;
    let dxgi: IMFDXGIBuffer = buffer.cast().ok()?;
    let mut p = std::ptr::null_mut();
    // SAFETY: valid out-pointer; a success transfers one reference.
    unsafe { dxgi.GetResource(&ID3D11Texture2D::IID, &mut p) }.ok()?;
    // SAFETY: `p` is an `ID3D11Texture2D` whose reference this takes.
    let texture = unsafe { ID3D11Texture2D::from_raw(p) };
    // SAFETY: a live buffer.
    let subresource = unsafe { dxgi.GetSubresourceIndex() }.ok()?;
    Some((texture, subresource))
}

type FnStartup = unsafe extern "system" fn(u32, u32) -> HRESULT;
type FnShutdown = unsafe extern "system" fn() -> HRESULT;
type FnTEnumEx = unsafe extern "system" fn(
    GUID,
    u32,
    *const MFT_REGISTER_TYPE_INFO,
    *const MFT_REGISTER_TYPE_INFO,
    *mut *mut *mut c_void,
    *mut u32,
) -> HRESULT;
type FnCreateObject = unsafe extern "system" fn(*mut *mut c_void) -> HRESULT;
type FnCreateMemoryBuffer = unsafe extern "system" fn(u32, *mut *mut c_void) -> HRESULT;
type FnCreateDxgiDeviceManager = unsafe extern "system" fn(*mut u32, *mut *mut c_void) -> HRESULT;
type FnCreateDxgiSurfaceBuffer =
    unsafe extern "system" fn(*const GUID, *mut c_void, u32, i32, *mut *mut c_void) -> HRESULT;

/// The `mfplat.dll` entry points the codecs use.
struct Api {
    startup: FnStartup,
    shutdown: FnShutdown,
    tenum_ex: FnTEnumEx,
    create_media_type: FnCreateObject,
    create_sample: FnCreateObject,
    create_memory_buffer: FnCreateMemoryBuffer,
    create_dxgi_device_manager: FnCreateDxgiDeviceManager,
    create_dxgi_surface_buffer: FnCreateDxgiSurfaceBuffer,
}

static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();

fn api() -> Result<&'static Api> {
    API.get_or_init(load)
        .as_ref()
        .map_err(|e| CodecError::NoDevice(e.clone()))
}

fn load() -> std::result::Result<Api, String> {
    // SAFETY: loads a system library from System32 only; its initialisers are
    // those every Media Foundation client runs.
    let lib = unsafe { Library::load_with_flags("mfplat.dll", LOAD_LIBRARY_SEARCH_SYSTEM32) }
        .map_err(|e| {
            format!(
                "Media Foundation is not installed (mfplat.dll: {e}); \
                 Windows N editions need the Media Feature Pack"
            )
        })?;
    // The library stays loaded for the life of the process, so the
    // function pointers below never dangle.
    let lib: &'static Library = Box::leak(Box::new(lib));
    fn get<T: Copy>(lib: &'static Library, name: &str) -> std::result::Result<T, String> {
        let symbol = format!("{name}\0");
        // SAFETY: each name is resolved to the function type its SDK
        // declaration has.
        unsafe { lib.get::<T>(symbol.as_bytes()) }
            .map(|s| *s)
            .map_err(|e| format!("mfplat.dll has no {name}: {e}"))
    }
    Ok(Api {
        startup: get(lib, "MFStartup")?,
        shutdown: get(lib, "MFShutdown")?,
        tenum_ex: get(lib, "MFTEnumEx")?,
        create_media_type: get(lib, "MFCreateMediaType")?,
        create_sample: get(lib, "MFCreateSample")?,
        create_memory_buffer: get(lib, "MFCreateMemoryBuffer")?,
        create_dxgi_device_manager: get(lib, "MFCreateDXGIDeviceManager")?,
        create_dxgi_surface_buffer: get(lib, "MFCreateDXGISurfaceBuffer")?,
    })
}

/// The calling thread's membership of a COM apartment.
struct Apartment(bool);

impl Drop for Apartment {
    fn drop(&mut self) {
        if self.0 {
            // SAFETY: balances the successful `CoInitializeEx` on this thread.
            unsafe { CoUninitialize() };
        }
    }
}

thread_local! {
    static APARTMENT: Apartment = Apartment(
        // SAFETY: no reserved pointer. Success, including `S_FALSE` for a
        // thread already in the apartment, is balanced when the thread exits.
        unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) }.is_ok(),
    );
}

/// Joins the calling thread to COM's multithreaded apartment, once per
/// thread. A thread already in a single-threaded apartment keeps it; the
/// transforms and buffers used here are free-threaded.
pub(crate) fn com() {
    APARTMENT.with(|_| {});
}

/// A hold on the Media Foundation platform, which stays started while any
/// hold lives.
pub(crate) struct Platform(&'static Api);

impl Platform {
    pub(crate) fn start() -> Result<Self> {
        let api = api()?;
        com();
        // SAFETY: plain start-up call, balanced by `MFShutdown` on drop.
        unsafe { (api.startup)(MF_VERSION, MFSTARTUP_LITE) }
            .ok()
            .map_err(|e| error(e, "MFStartup"))?;
        Ok(Self(api))
    }

    /// A new, empty media type.
    pub(crate) fn media_type(&self) -> Result<IMFMediaType> {
        let mut p = std::ptr::null_mut();
        // SAFETY: valid out-pointer; a success transfers one reference.
        unsafe { (self.0.create_media_type)(&mut p) }
            .ok()
            .map_err(|e| error(e, "MFCreateMediaType"))?;
        // SAFETY: `p` is an `IMFMediaType` whose reference this takes.
        Ok(unsafe { IMFMediaType::from_raw(p) })
    }

    /// A sample holding one system-memory buffer of `len` bytes, with its
    /// current length 0.
    pub(crate) fn sample(&self, len: usize) -> Result<(IMFSample, IMFMediaBuffer)> {
        let len = u32::try_from(len)
            .map_err(|_| CodecError::InvalidInput(format!("a {len}-byte buffer is too large")))?;
        let mut p = std::ptr::null_mut();
        // SAFETY: valid out-pointer; a success transfers one reference.
        unsafe { (self.0.create_sample)(&mut p) }
            .ok()
            .map_err(|e| error(e, "MFCreateSample"))?;
        // SAFETY: `p` is an `IMFSample` whose reference this takes.
        let sample = unsafe { IMFSample::from_raw(p) };
        let mut p = std::ptr::null_mut();
        // SAFETY: as above.
        unsafe { (self.0.create_memory_buffer)(len, &mut p) }
            .ok()
            .map_err(|e| error(e, "MFCreateMemoryBuffer"))?;
        // SAFETY: `p` is an `IMFMediaBuffer` whose reference this takes.
        let buffer = unsafe { IMFMediaBuffer::from_raw(p) };
        // SAFETY: live objects.
        unsafe { sample.AddBuffer(&buffer) }.map_err(|e| error(e, "IMFSample::AddBuffer"))?;
        Ok((sample, buffer))
    }

    /// A DXGI device manager over HAL's process D3D11 device, for transforms
    /// that take D3D11 textures. `None` when there is no HAL device.
    pub(crate) fn device_manager(&self) -> Option<IMFDXGIDeviceManager> {
        let device = hal_device().ok()?;
        let mut token = 0u32;
        let mut p = std::ptr::null_mut();
        // SAFETY: valid out-pointers; a success transfers one reference.
        if let Err(e) = unsafe { (self.0.create_dxgi_device_manager)(&mut token, &mut p) }.ok() {
            log::debug!("MFCreateDXGIDeviceManager: {e}");
            return None;
        }
        // SAFETY: `p` is an `IMFDXGIDeviceManager` whose reference this takes.
        let manager = unsafe { IMFDXGIDeviceManager::from_raw(p) };
        // SAFETY: live objects; the token is the one the manager returned.
        match unsafe { manager.ResetDevice(&device, token) } {
            Ok(()) => Some(manager),
            Err(e) => {
                log::debug!("IMFDXGIDeviceManager::ResetDevice: {e}");
                None
            }
        }
    }

    /// A sample over subresource 0 of `texture`, which Media Foundation reads
    /// in place.
    pub(crate) fn texture_sample(&self, texture: &ID3D11Texture2D) -> Result<IMFSample> {
        let mut p = std::ptr::null_mut();
        // SAFETY: valid out-pointer; the buffer takes its own reference to
        // the live texture.
        unsafe {
            (self.0.create_dxgi_surface_buffer)(
                &ID3D11Texture2D::IID,
                texture.as_raw(),
                0,
                0,
                &mut p,
            )
        }
        .ok()
        .map_err(|e| error(e, "MFCreateDXGISurfaceBuffer"))?;
        // SAFETY: `p` is an `IMFMediaBuffer` whose reference this takes.
        let buffer = unsafe { IMFMediaBuffer::from_raw(p) };
        let mut p = std::ptr::null_mut();
        // SAFETY: valid out-pointer; a success transfers one reference.
        unsafe { (self.0.create_sample)(&mut p) }
            .ok()
            .map_err(|e| error(e, "MFCreateSample"))?;
        // SAFETY: `p` is an `IMFSample` whose reference this takes.
        let sample = unsafe { IMFSample::from_raw(p) };
        // SAFETY: live objects.
        unsafe { sample.AddBuffer(&buffer) }.map_err(|e| error(e, "IMFSample::AddBuffer"))?;
        Ok(sample)
    }

    /// The transforms that convert video `input` (any subtype when `None`)
    /// to `output` in `category`: hardware transforms first unless
    /// `hardware` is false, then software ones, each in the order Media
    /// Foundation ranks them. Setting the environment variable `env` keeps
    /// only transforms whose friendly name contains its value, ignoring
    /// case; the values `hardware` and `software` keep only that kind.
    pub(crate) fn transforms(
        &self,
        category: GUID,
        input: Option<GUID>,
        output: GUID,
        hardware: bool,
        env: &str,
    ) -> Result<Vec<Candidate>> {
        let filter = std::env::var(env).ok().filter(|v| !v.is_empty());
        let input = input.map(|subtype| MFT_REGISTER_TYPE_INFO {
            guidMajorType: MFMediaType_Video,
            guidSubtype: subtype,
        });
        let output = MFT_REGISTER_TYPE_INFO {
            guidMajorType: MFMediaType_Video,
            guidSubtype: output,
        };
        let mut passes = vec![(
            MFT_ENUM_FLAG_SYNCMFT.0 | MFT_ENUM_FLAG_SORTANDFILTER.0,
            false,
        )];
        if hardware || filter.is_some() {
            let flags =
                MFT_ENUM_FLAG_HARDWARE.0 | MFT_ENUM_FLAG_ASYNCMFT.0 | MFT_ENUM_FLAG_SORTANDFILTER.0;
            passes.insert(0, (flags, true));
        }
        let mut found = Vec::new();
        for (flags, is_hardware) in passes {
            let mut array = std::ptr::null_mut();
            let mut count = 0u32;
            // SAFETY: valid type infos and out-pointers.
            let enumerated = unsafe {
                (self.0.tenum_ex)(
                    category,
                    flags as u32,
                    input.as_ref().map_or(std::ptr::null(), |i| i as *const _),
                    &output,
                    &mut array,
                    &mut count,
                )
            };
            if let Err(e) = enumerated.ok() {
                log::debug!("MFTEnumEx({flags:#x}): {e}");
                continue;
            }
            for i in 0..count as usize {
                // SAFETY: `array` holds `count` activation objects, each with
                // a reference this takes.
                let raw = unsafe { *array.add(i) };
                if raw.is_null() {
                    continue;
                }
                // SAFETY: as above.
                let activate = unsafe { IMFActivate::from_raw(raw) };
                let name = string(
                    &activate.cast::<IMFAttributes>().ok(),
                    &MFT_FRIENDLY_NAME_Attribute,
                )
                .unwrap_or_else(|| "unnamed transform".to_owned());
                found.push(Candidate {
                    activate,
                    name,
                    hardware: is_hardware,
                });
            }
            if !array.is_null() {
                // SAFETY: the array was allocated by `MFTEnumEx` with the COM
                // allocator; its elements were taken above.
                unsafe { CoTaskMemFree(Some(array as *const c_void)) };
            }
        }
        Ok(match filter {
            Some(f) => found.into_iter().filter(|c| c.matches(&f)).collect(),
            None => found,
        })
    }
}

impl Drop for Platform {
    fn drop(&mut self) {
        com();
        // SAFETY: balances the `MFStartup` of this hold.
        let _ = unsafe { (self.0.shutdown)() };
    }
}

/// A transform Media Foundation enumerated, not yet activated.
pub(crate) struct Candidate {
    pub activate: IMFActivate,
    pub name: String,
    pub hardware: bool,
}

impl Candidate {
    /// Whether the environment filter `filter` keeps this transform.
    fn matches(&self, filter: &str) -> bool {
        name_matches(&self.name, self.hardware, filter)
    }
}

/// Whether a transform named `name` passes the environment filter `filter`.
fn name_matches(name: &str, hardware: bool, filter: &str) -> bool {
    match filter.to_ascii_lowercase().as_str() {
        "hardware" => hardware,
        "software" => !hardware,
        f => name.to_ascii_lowercase().contains(f),
    }
}

/// Maps a Media Foundation failure of `op` to [`CodecError::Device`].
pub(crate) fn error(e: windows::core::Error, op: &str) -> CodecError {
    let message = e.message();
    let text = if message.is_empty() {
        format!("HRESULT {:#010x}", e.code().0)
    } else {
        format!("{message} ({:#010x})", e.code().0)
    };
    CodecError::Device {
        op: op.to_owned(),
        source: std::io::Error::other(text),
    }
}

/// A string attribute, or `None` when it is not set.
fn string(attributes: &Option<IMFAttributes>, key: &GUID) -> Option<String> {
    let attributes = attributes.as_ref()?;
    let mut p = PWSTR::null();
    let mut len = 0;
    // SAFETY: valid out-pointers; the string is freed below.
    unsafe { attributes.GetAllocatedString(key, &mut p, &mut len) }.ok()?;
    // SAFETY: Media Foundation returned a NUL-terminated string.
    let s = unsafe { p.to_string() }.ok();
    // SAFETY: allocated by `GetAllocatedString` with the COM allocator.
    unsafe { CoTaskMemFree(Some(p.0 as *const c_void)) };
    s
}

/// Sets a size or ratio attribute, which Media Foundation packs as two
/// 32-bit halves of a 64-bit value.
pub(crate) fn set_pair(t: &IMFAttributes, key: &GUID, (hi, lo): (u32, u32)) -> Result<()> {
    // SAFETY: a live attribute store.
    unsafe { t.SetUINT64(key, (u64::from(hi) << 32) | u64::from(lo)) }
        .map_err(|e| error(e, "IMFAttributes::SetUINT64"))
}

/// Reads a size or ratio attribute set with [`set_pair`].
pub(crate) fn pair(t: &IMFAttributes, key: &GUID) -> Option<(u32, u32)> {
    // SAFETY: a live attribute store.
    let v = unsafe { t.GetUINT64(key) }.ok()?;
    Some(((v >> 32) as u32, v as u32))
}

/// A frame rate as the ratio Media Foundation takes: exact for whole rates
/// and the NTSC rates (N * 1000 / 1001), in thousandths otherwise.
pub(crate) fn frame_rate_ratio(fps: f64) -> (u32, u32) {
    let whole = fps.round();
    if (fps - whole).abs() < 1e-6 {
        return (whole as u32, 1);
    }
    let ntsc = (fps * 1.001).round();
    if (fps - ntsc * 1000.0 / 1001.0).abs() < 5e-3 {
        return ((ntsc as u32) * 1000, 1001);
    }
    ((fps * 1000.0).round() as u32, 1000)
}

/// The length of one frame at `fps`, in the 100 ns units of sample times.
pub(crate) fn frame_duration(fps: f64) -> i64 {
    let (num, den) = frame_rate_ratio(fps);
    (10_000_000 * i64::from(den) / i64::from(num.max(1))).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_rates_are_exact_where_media_foundation_expects_it() {
        assert_eq!(frame_rate_ratio(30.0), (30, 1));
        assert_eq!(frame_rate_ratio(60.0), (60, 1));
        assert_eq!(frame_rate_ratio(30000.0 / 1001.0), (30000, 1001));
        assert_eq!(frame_rate_ratio(29.97), (30000, 1001));
        assert_eq!(frame_rate_ratio(23.976), (24000, 1001));
        assert_eq!(frame_rate_ratio(12.5), (12500, 1000));
    }

    #[test]
    fn frame_durations_are_in_100_ns_units() {
        assert_eq!(frame_duration(30.0), 333_333);
        assert_eq!(frame_duration(25.0), 400_000);
        assert_eq!(frame_duration(29.97), 333_666);
    }

    #[test]
    fn the_environment_filter_matches_names_or_kinds() {
        assert!(name_matches("NVIDIA H.264 Encoder MFT", true, "nvidia"));
        assert!(name_matches("NVIDIA H.264 Encoder MFT", true, "hardware"));
        assert!(!name_matches("NVIDIA H.264 Encoder MFT", true, "software"));
        assert!(name_matches("H264 Encoder MFT", false, "SOFTWARE"));
        assert!(!name_matches("H264 Encoder MFT", false, "intel"));
    }

    #[test]
    fn media_foundation_is_present_or_reported_as_no_device() {
        match Platform::start() {
            Ok(p) => {
                let t = p.media_type().expect("an empty media type");
                set_pair(
                    &t.cast().unwrap(),
                    &windows::Win32::Media::MediaFoundation::MF_MT_FRAME_SIZE,
                    (1920, 1080),
                )
                .unwrap();
                assert_eq!(
                    pair(
                        &t.cast().unwrap(),
                        &windows::Win32::Media::MediaFoundation::MF_MT_FRAME_SIZE
                    ),
                    Some((1920, 1080))
                );
                let (sample, buffer) = p.sample(64).expect("a sample");
                // SAFETY: live objects.
                assert_eq!(unsafe { sample.GetBufferCount() }.unwrap(), 1);
                // SAFETY: as above.
                assert_eq!(unsafe { buffer.GetMaxLength() }.unwrap(), 64);
            }
            Err(CodecError::NoDevice(why)) => {
                eprintln!("WARNING: SKIPPED: media_foundation_is_present - {why}")
            }
            Err(e) => panic!("{e}"),
        }
    }
}
