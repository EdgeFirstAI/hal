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
//! Pixels move through system memory. Source tensors are read and copied into
//! Media Foundation buffers, and decoded pictures are copied into the
//! decoder's pool: D3D11 tensors where the process has a D3D11 device, host
//! memory otherwise. The transforms' own textures cannot be wrapped as
//! tensors: they are `DXGI_FORMAT_NV12`, while an NV12 D3D11 tensor is an
//! `R8` texture 1.5 times the picture height.
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

use libloading::os::windows::{Library, LOAD_LIBRARY_SEARCH_SYSTEM32};
use windows::core::{Interface, GUID, HRESULT, PWSTR};
use windows::Win32::Media::MediaFoundation::{
    IMFActivate, IMFAttributes, IMFMediaBuffer, IMFMediaType, IMFSample, MFMediaType_Video,
    MFT_FRIENDLY_NAME_Attribute, MFSTARTUP_LITE, MFT_ENUM_FLAG_ASYNCMFT, MFT_ENUM_FLAG_HARDWARE,
    MFT_ENUM_FLAG_SORTANDFILTER, MFT_ENUM_FLAG_SYNCMFT, MFT_REGISTER_TYPE_INFO, MF_VERSION,
};
use windows::Win32::System::Com::{
    CoInitializeEx, CoTaskMemFree, CoUninitialize, COINIT_MULTITHREADED,
};

use crate::{CodecError, Result};

pub(crate) use transform::{Output, Transform};

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

/// The `mfplat.dll` entry points the codecs use.
struct Api {
    startup: FnStartup,
    shutdown: FnShutdown,
    tenum_ex: FnTEnumEx,
    create_media_type: FnCreateObject,
    create_sample: FnCreateObject,
    create_memory_buffer: FnCreateMemoryBuffer,
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
