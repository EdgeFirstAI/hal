// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Codecs, engines, and what the engines on this machine can do.
//!
//! The enums are the C names of the values; entry points take them as
//! `uint32_t` and refuse a value outside the enum with `EINVAL`.

use std::ffi::{c_char, c_int, CStr, CString};

use edgefirst_codec::{Capability, Codec, Direction, Engine, Transfer};

use crate::error::{guard, invalid, set_last};

/// A compressed image or video format.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfCodec {
    /// JPEG still images.
    Jpeg = 0,
    /// PNG still images.
    Png = 1,
    /// H.264, as an Annex B byte stream.
    H264 = 2,
    /// HEVC, as an Annex B byte stream. No engine runs it yet.
    Hevc = 3,
    /// Motion JPEG: a stream of JPEG pictures.
    Mjpeg = 4,
    /// VP8. No engine runs it yet.
    Vp8 = 5,
    /// VP9. No engine runs it yet.
    Vp9 = 6,
    /// AV1. No engine runs it yet.
    Av1 = 7,
}

/// What runs a codec.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfCodecEngine {
    /// The first engine that can do the request, in the platform's order.
    /// The environment variable `EDGEFIRST_CODEC_ENGINE` (engine names,
    /// comma-separated) narrows and orders the choice.
    Auto = 0,
    /// Linux V4L2 memory-to-memory devices.
    V4l2 = 1,
    /// Windows Media Foundation transforms.
    MediaFoundation = 2,
    /// Apple VideoToolbox.
    Videotoolbox = 3,
    /// NVIDIA nvJPEG, on Linux with CUDA.
    Nvjpeg = 4,
    /// HAL's own CPU code, for still images only.
    Cpu = 5,
}

/// Encoding or decoding.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfCodecDirection {
    /// Pixels to a compressed format.
    Encode = 0,
    /// A compressed format to pixels.
    Decode = 1,
}

/// How pixels move between a tensor and an engine.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfCodecTransfer {
    /// The engine reads or writes the tensor's own memory.
    ZeroCopy = 0,
    /// One copy per frame on the GPU.
    GpuCopy = 1,
    /// One copy per frame through CPU memory.
    CpuCopy = 2,
}

pub(crate) fn codec_from(code: u32) -> Option<Codec> {
    Some(match code {
        0 => Codec::Jpeg,
        1 => Codec::Png,
        2 => Codec::H264,
        3 => Codec::Hevc,
        4 => Codec::Mjpeg,
        5 => Codec::Vp8,
        6 => Codec::Vp9,
        7 => Codec::Av1,
        _ => return None,
    })
}

pub(crate) fn codec_code(codec: Codec) -> u32 {
    (match codec {
        Codec::Jpeg => EfCodec::Jpeg,
        Codec::Png => EfCodec::Png,
        Codec::H264 => EfCodec::H264,
        Codec::Hevc => EfCodec::Hevc,
        Codec::Mjpeg => EfCodec::Mjpeg,
        Codec::Vp8 => EfCodec::Vp8,
        Codec::Vp9 => EfCodec::Vp9,
        Codec::Av1 => EfCodec::Av1,
        _ => return u32::MAX,
    }) as u32
}

pub(crate) fn engine_from(code: u32) -> Option<Engine> {
    Some(match code {
        0 => Engine::Auto,
        1 => Engine::V4l2,
        2 => Engine::MediaFoundation,
        3 => Engine::VideoToolbox,
        4 => Engine::NvJpeg,
        5 => Engine::Cpu,
        _ => return None,
    })
}

pub(crate) fn engine_code(engine: Engine) -> u32 {
    (match engine {
        Engine::Auto => EfCodecEngine::Auto,
        Engine::V4l2 => EfCodecEngine::V4l2,
        Engine::MediaFoundation => EfCodecEngine::MediaFoundation,
        Engine::VideoToolbox => EfCodecEngine::Videotoolbox,
        Engine::NvJpeg => EfCodecEngine::Nvjpeg,
        Engine::Cpu => EfCodecEngine::Cpu,
        _ => return u32::MAX,
    }) as u32
}

pub(crate) fn transfer_code(transfer: Transfer) -> u32 {
    (match transfer {
        Transfer::ZeroCopy => EfCodecTransfer::ZeroCopy,
        Transfer::GpuCopy => EfCodecTransfer::GpuCopy,
        Transfer::CpuCopy => EfCodecTransfer::CpuCopy,
        _ => return u32::MAX,
    }) as u32
}

fn direction_code(direction: Direction) -> u32 {
    (match direction {
        Direction::Encode => EfCodecDirection::Encode,
        Direction::Decode => EfCodecDirection::Decode,
    }) as u32
}

/// The engine `code` names, or `EINVAL` recorded as the last error.
pub(crate) fn engine_arg(code: u32) -> Result<Engine, c_int> {
    engine_from(code).ok_or_else(|| invalid(&format!("{code} is not an ef_codec_engine")))
}

/// The codec `code` names, or `EINVAL` recorded as the last error.
pub(crate) fn codec_arg(code: u32) -> Result<Codec, c_int> {
    codec_from(code).ok_or_else(|| invalid(&format!("{code} is not an ef_codec")))
}

/// The usual name of an `ef_codec` value (`"H.264"`, `"JPEG"`), or `NULL`
/// for a value outside the enum. The string is static.
#[no_mangle]
pub extern "C" fn ef_codec_name(codec: u32) -> *const c_char {
    let name: &CStr = match codec_from(codec) {
        Some(Codec::Jpeg) => c"JPEG",
        Some(Codec::Png) => c"PNG",
        Some(Codec::H264) => c"H.264",
        Some(Codec::Hevc) => c"HEVC",
        Some(Codec::Mjpeg) => c"MJPEG",
        Some(Codec::Vp8) => c"VP8",
        Some(Codec::Vp9) => c"VP9",
        Some(Codec::Av1) => c"AV1",
        _ => return std::ptr::null(),
    };
    name.as_ptr()
}

/// The name of an `ef_codec_engine` value, as `EDGEFIRST_CODEC_ENGINE` and
/// [`ef_codec_engine_from_name`] take it (`"v4l2"`, `"media-foundation"`),
/// or `NULL` for a value outside the enum. The string is static.
#[no_mangle]
pub extern "C" fn ef_codec_engine_name(engine: u32) -> *const c_char {
    let name: &CStr = match engine_from(engine) {
        Some(Engine::Auto) => c"auto",
        Some(Engine::V4l2) => c"v4l2",
        Some(Engine::MediaFoundation) => c"media-foundation",
        Some(Engine::VideoToolbox) => c"videotoolbox",
        Some(Engine::NvJpeg) => c"nvjpeg",
        Some(Engine::Cpu) => c"cpu",
        _ => return std::ptr::null(),
    };
    name.as_ptr()
}

/// Parses an engine name, ignoring case (`"mf"` and `"vt"` are accepted for
/// Media Foundation and VideoToolbox), into `*out`.
///
/// @return 0, or `EINVAL` for an unknown name.
///
/// # Safety
/// `name` must be a NUL-terminated string and `out` writable.
#[no_mangle]
pub unsafe extern "C" fn ef_codec_engine_from_name(name: *const c_char, out: *mut u32) -> c_int {
    guard(|| {
        if name.is_null() || out.is_null() {
            return invalid("name and out must not be NULL");
        }
        let Ok(name) = unsafe { CStr::from_ptr(name) }.to_str() else {
            return invalid("the engine name is not UTF-8");
        };
        match name.parse::<Engine>() {
            Ok(engine) => {
                unsafe { *out = engine_code(engine) };
                0
            }
            Err(e) => {
                set_last(&e.to_string());
                libc::EINVAL
            }
        }
    })
}

/// One thing an engine on this machine can do.
#[repr(C)]
pub struct EfCodecCapability {
    /// The engine, an `ef_codec_engine`.
    pub engine: u32,
    /// The compressed format, an `ef_codec`.
    pub codec: u32,
    /// An `ef_codec_direction`.
    pub direction: u32,
    /// How frames move in the best case (an `ef_codec_transfer`), for a
    /// source or pool in the memory the engine prefers. An open stream says
    /// what it does in its own info.
    pub transfer: u32,
    /// The work runs on dedicated hardware rather than the CPU.
    pub hardware: bool,
    /// The device within the engine: a device node, a transform's name, or
    /// `"cpu"`.
    pub device: *const c_char,
    /// The pixel formats of the images (`"NV12"`, `"YUYV"`): what an encoder
    /// takes, or what a decoder produces.
    pub formats: *const *const c_char,
    /// Entries in `formats`.
    pub format_count: usize,
}

/// The strings an [`EfCodecCapability`] points into.
struct Strings {
    device: CString,
    _formats: Vec<CString>,
    format_ptrs: Vec<*const c_char>,
}

/// What the codec engines on this machine can do: a list of
/// `ef_codec_capability`, engine by engine, in the order
/// `EF_CODEC_ENGINE_AUTO` tries them.
pub struct EfCodecCapabilities {
    items: Vec<EfCodecCapability>,
    _strings: Vec<Strings>,
}

impl EfCodecCapabilities {
    fn from(found: Vec<Capability>) -> Self {
        let strings: Vec<Strings> = found
            .iter()
            .map(|c| {
                let formats: Vec<CString> = c
                    .formats
                    .iter()
                    .map(|f| CString::new(f.as_str()).unwrap_or_default())
                    .collect();
                let format_ptrs = formats.iter().map(|f| f.as_ptr()).collect();
                Strings {
                    device: CString::new(c.device.replace('\0', "?")).unwrap_or_default(),
                    _formats: formats,
                    format_ptrs,
                }
            })
            .collect();
        // The pointers are to the strings' heap buffers, which stay put when
        // the vectors holding them move into the list.
        let items = found
            .iter()
            .zip(&strings)
            .map(|(c, s)| EfCodecCapability {
                engine: engine_code(c.engine),
                codec: codec_code(c.codec),
                direction: direction_code(c.direction),
                transfer: transfer_code(c.transfer),
                hardware: c.hardware,
                device: s.device.as_ptr(),
                formats: s.format_ptrs.as_ptr(),
                format_count: s.format_ptrs.len(),
            })
            .collect();
        Self {
            items,
            _strings: strings,
        }
    }
}

/// Probes the codec engines on this machine.
///
/// Probing opens no stream, but it opens device nodes and enumerates (on
/// Windows, activates) transforms, so it is not free: call it once.
///
/// @return the list, freed with [`ef_codec_capabilities_free`]; `NULL` only
///         if allocation fails.
#[no_mangle]
pub extern "C" fn ef_codec_capabilities_new() -> *mut EfCodecCapabilities {
    crate::error::guard_new(|| Ok(EfCodecCapabilities::from(edgefirst_codec::capabilities())))
}

/// Entries in the list; 0 for `NULL`.
///
/// # Safety
/// `list` must be `NULL` or a live list.
#[no_mangle]
pub unsafe extern "C" fn ef_codec_capabilities_count(list: *const EfCodecCapabilities) -> usize {
    unsafe { list.as_ref() }.map_or(0, |l| l.items.len())
}

/// Entry `index`, valid until the list is freed, or `NULL` past the end.
///
/// # Safety
/// `list` must be `NULL` or a live list.
#[no_mangle]
pub unsafe extern "C" fn ef_codec_capabilities_get(
    list: *const EfCodecCapabilities,
    index: usize,
) -> *const EfCodecCapability {
    unsafe { list.as_ref() }
        .and_then(|l| l.items.get(index))
        .map_or(std::ptr::null(), |c| c as *const _)
}

/// Frees a list. Freeing `NULL` is a no-op.
///
/// # Safety
/// `list` must be `NULL` or have come from [`ef_codec_capabilities_new`].
#[no_mangle]
pub unsafe extern "C" fn ef_codec_capabilities_free(list: *mut EfCodecCapabilities) {
    if !list.is_null() {
        drop(unsafe { Box::from_raw(list) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use edgefirst_tensor::PixelFormat;

    #[test]
    fn every_code_round_trips() {
        for code in 0..8 {
            assert_eq!(codec_code(codec_from(code).unwrap()), code);
            assert!(!ef_codec_name(code).is_null());
        }
        assert!(codec_from(8).is_none());
        assert!(ef_codec_name(8).is_null());
        for code in 0..6 {
            let engine = engine_from(code).unwrap();
            assert_eq!(engine_code(engine), code);
            let name = unsafe { CStr::from_ptr(ef_codec_engine_name(code)) };
            assert_eq!(name.to_str().unwrap(), engine.name());
            let mut parsed = u32::MAX;
            assert_eq!(
                unsafe { ef_codec_engine_from_name(name.as_ptr(), &mut parsed) },
                0
            );
            assert_eq!(parsed, code);
        }
        assert!(engine_from(6).is_none());
        let mut parsed = 0;
        assert_eq!(
            unsafe { ef_codec_engine_from_name(c"warp".as_ptr(), &mut parsed) },
            libc::EINVAL
        );
    }

    /// The CPU JPEG encoder is on every machine, so the probe always has an
    /// entry whose strings can be read back.
    #[test]
    fn the_probe_lists_the_cpu_jpeg_encoder_with_its_formats() {
        let list = ef_codec_capabilities_new();
        assert!(!list.is_null());
        unsafe {
            assert!(ef_codec_capabilities_get(list, ef_codec_capabilities_count(list)).is_null());
            let cpu = (0..ef_codec_capabilities_count(list))
                .map(|i| &*ef_codec_capabilities_get(list, i))
                .find(|c| {
                    c.engine == EfCodecEngine::Cpu as u32
                        && c.codec == EfCodec::Jpeg as u32
                        && c.direction == EfCodecDirection::Encode as u32
                })
                .expect("the CPU JPEG encoder is listed");
            assert!(!cpu.hardware);
            assert_eq!(cpu.transfer, EfCodecTransfer::ZeroCopy as u32);
            assert_eq!(CStr::from_ptr(cpu.device).to_str().unwrap(), "cpu");
            let formats: Vec<&str> = std::slice::from_raw_parts(cpu.formats, cpu.format_count)
                .iter()
                .map(|&f| CStr::from_ptr(f).to_str().unwrap())
                .collect();
            assert!(formats.contains(&PixelFormat::Nv12.as_str()), "{formats:?}");
            ef_codec_capabilities_free(list);
        }
    }
}
