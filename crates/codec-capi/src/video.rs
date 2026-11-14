// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Video encoding and decoding: tensors in, access units out, and back.
//!
//! Both directions push work in and pull results out. An encoder takes
//! source frames and queues finished access units, which the caller pulls
//! with [`ef_video_encoder_next_packet`]; a decoder takes access units and
//! queues decoded frames and format changes, pulled with
//! [`ef_video_decoder_next_event`]. Each object is used from one thread at a
//! time.

use std::collections::VecDeque;
use std::ffi::{c_char, c_int, CStr, CString};

use edgefirst_codec::video::{
    Bitrate, DecodeEvent, DecodedFrame, DecoderConfig, EncodedFrame, EncoderConfig, FrameOptions,
    H264Level, H264Profile, OutputMemory, OutputPool, StreamInfo, VideoDecoder, VideoEncoder,
};
use edgefirst_tensor::{PixelFormat, Region, TensorDyn};
use edgefirst_tensor_ffi::EfTensor;

use crate::engine::{codec_arg, engine_arg, engine_code, transfer_code};
use crate::error::{fail, guard, guard_new, invalid};

/// Encode the frame as a key frame (an IDR picture); a flag of
/// [`ef_video_encoder_encode`].
pub const EF_VIDEO_ENCODE_KEYFRAME: u32 = 1;

/// A rectangle of an image, in pixels.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EfCodecRegion {
    /// Left edge.
    pub x: u32,
    /// Top edge.
    pub y: u32,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
}

impl From<Region> for EfCodecRegion {
    fn from(r: Region) -> Self {
        let c = |v: usize| u32::try_from(v).unwrap_or(u32::MAX);
        Self {
            x: c(r.x),
            y: c(r.y),
            width: c(r.width),
            height: c(r.height),
        }
    }
}

/// H.264 profile of an encoder.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfH264Profile {
    /// High where the device offers it, else the device default.
    Auto = 0,
    /// Baseline.
    Baseline = 1,
    /// Constrained Baseline.
    ConstrainedBaseline = 2,
    /// Main.
    Main = 3,
    /// High.
    High = 4,
}

/// What [`ef_video_decoder_next_event`] returned.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfVideoEvent {
    /// Nothing is ready: decode more, or flush.
    None = 0,
    /// A decoded frame, in presentation order.
    Frame = 1,
    /// The stream's size or format changed, or became known; read it with
    /// [`ef_video_decoder_stream_info`]. Frames after this have the new
    /// format; frames already returned keep their memory.
    Reconfigured = 2,
}

/// Whose memory a decoder writes its pictures to.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfVideoMemory {
    /// The pool's tensors, which the device imports, or which each picture
    /// is copied into.
    Imported = 0,
    /// The device's own buffers, exported as DMA-BUFs because it would not
    /// import the pool's. CPU reads of these may be uncached and slow.
    Exported = 1,
}

/// A pixel format argument, or `EINVAL` recorded as the last error.
///
/// # Safety
/// `format` must be `NULL` or a NUL-terminated string.
unsafe fn format_arg(format: *const c_char) -> Result<PixelFormat, c_int> {
    if format.is_null() {
        return Err(invalid("the pixel format is NULL"));
    }
    let s = unsafe { CStr::from_ptr(format) }.to_string_lossy();
    PixelFormat::from_str_code(&s).ok_or_else(|| invalid(&format!("{s:?} is not a pixel format")))
}

/// The level `code` names: the level number times ten, 9 for level 1b.
fn level_from(code: u32) -> Option<H264Level> {
    use H264Level::*;
    Some(match code {
        9 => L1b,
        10 => L1_0,
        11 => L1_1,
        12 => L1_2,
        13 => L1_3,
        20 => L2_0,
        21 => L2_1,
        22 => L2_2,
        30 => L3_0,
        31 => L3_1,
        32 => L3_2,
        40 => L4_0,
        41 => L4_1,
        42 => L4_2,
        50 => L5_0,
        51 => L5_1,
        52 => L5_2,
        60 => L6_0,
        61 => L6_1,
        62 => L6_2,
        _ => return None,
    })
}

/// A CString of `s`, with any NUL replaced.
fn c_string(s: &str) -> CString {
    CString::new(s.replace('\0', "?")).unwrap_or_default()
}

/// The configuration of a video encoder.
pub struct EfVideoEncoderConfig {
    inner: EncoderConfig,
}

/// Creates an encoder configuration for `codec` (an `ef_codec`): sources of
/// `width` x `height` pixels in pixel format `format` (`"NV12"`, `"YUYV"`)
/// at a nominal `frame_rate`, on `EF_CODEC_ENGINE_AUTO`, with an automatic
/// bitrate, profile and level, the device's default GOP, and the parameter
/// sets repeated on every key frame.
///
/// @return the configuration, freed with
///         [`ef_video_encoder_config_free`]; `NULL` with `errno` `EINVAL`
///         for a value outside `ef_codec` or an unknown pixel format.
///
/// # Safety
/// `format` must be a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn ef_video_encoder_config_new(
    codec: u32,
    width: u32,
    height: u32,
    format: *const c_char,
    frame_rate: f64,
) -> *mut EfVideoEncoderConfig {
    guard_new(|| {
        let codec = codec_arg(codec)?;
        let format = unsafe { format_arg(format) }?;
        Ok(EfVideoEncoderConfig {
            inner: EncoderConfig::new(codec, width, height, format, frame_rate),
        })
    })
}

/// Frees a configuration. Freeing `NULL` is a no-op.
///
/// # Safety
/// `c` must be `NULL` or have come from [`ef_video_encoder_config_new`].
#[no_mangle]
pub unsafe extern "C" fn ef_video_encoder_config_free(c: *mut EfVideoEncoderConfig) {
    if !c.is_null() {
        drop(unsafe { Box::from_raw(c) });
    }
}

/// Applies `set` to the configuration behind `c`.
///
/// # Safety
/// `c` must be `NULL` or a live configuration.
unsafe fn with_config(
    c: *mut EfVideoEncoderConfig,
    set: impl FnOnce(&mut EncoderConfig) -> Result<(), c_int>,
) -> c_int {
    guard(|| match unsafe { c.as_mut() } {
        Some(c) => set(&mut c.inner).err().unwrap_or(0),
        None => invalid("the configuration is NULL"),
    })
}

/// Selects the engine, an `ef_codec_engine`.
///
/// @return 0, or `EINVAL`.
///
/// # Safety
/// `c` must be a live configuration.
#[no_mangle]
pub unsafe extern "C" fn ef_video_encoder_config_set_engine(
    c: *mut EfVideoEncoderConfig,
    engine: u32,
) -> c_int {
    unsafe {
        with_config(c, |c| {
            c.engine = engine_arg(engine)?;
            Ok(())
        })
    }
}

/// Sets a constant target bitrate in bits per second; 0 chooses one from
/// the frame size and rate (0.16 bits per pixel, within 1 to 100 Mbps).
///
/// @return 0, or `EINVAL`.
///
/// # Safety
/// `c` must be a live configuration.
#[no_mangle]
pub unsafe extern "C" fn ef_video_encoder_config_set_bitrate(
    c: *mut EfVideoEncoderConfig,
    bps: u32,
) -> c_int {
    unsafe {
        with_config(c, |c| {
            c.bitrate = if bps == 0 {
                Bitrate::Auto
            } else {
                Bitrate::Bps(bps)
            };
            Ok(())
        })
    }
}

/// Sets the frames from one key frame to the next; 0 keeps the device
/// default.
///
/// @return 0, or `EINVAL`.
///
/// # Safety
/// `c` must be a live configuration.
#[no_mangle]
pub unsafe extern "C" fn ef_video_encoder_config_set_gop(
    c: *mut EfVideoEncoderConfig,
    gop: u32,
) -> c_int {
    unsafe {
        with_config(c, |c| {
            c.gop = (gop != 0).then_some(gop);
            Ok(())
        })
    }
}

/// Whether to emit the parameter sets with every key frame, so a receiver
/// can start at any key frame (on by default). The first frame always
/// carries them.
///
/// @return 0, or `EINVAL`.
///
/// # Safety
/// `c` must be a live configuration.
#[no_mangle]
pub unsafe extern "C" fn ef_video_encoder_config_set_repeat_headers(
    c: *mut EfVideoEncoderConfig,
    repeat: bool,
) -> c_int {
    unsafe {
        with_config(c, |c| {
            c.repeat_headers = repeat;
            Ok(())
        })
    }
}

/// Sets the H.264 profile, an `ef_h264_profile`.
///
/// @return 0, or `EINVAL`.
///
/// # Safety
/// `c` must be a live configuration.
#[no_mangle]
pub unsafe extern "C" fn ef_video_encoder_config_set_h264_profile(
    c: *mut EfVideoEncoderConfig,
    profile: u32,
) -> c_int {
    unsafe {
        with_config(c, |c| {
            c.profile = match profile {
                0 => None,
                1 => Some(H264Profile::Baseline),
                2 => Some(H264Profile::ConstrainedBaseline),
                3 => Some(H264Profile::Main),
                4 => Some(H264Profile::High),
                _ => return Err(invalid(&format!("{profile} is not an ef_h264_profile"))),
            };
            Ok(())
        })
    }
}

/// Sets the H.264 level: the level number times ten (`41` for 4.1), `9`
/// for level 1b, or 0 for the lowest level that covers the size, frame
/// rate and bitrate.
///
/// @return 0, or `EINVAL` for a level H.264 does not define.
///
/// # Safety
/// `c` must be a live configuration.
#[no_mangle]
pub unsafe extern "C" fn ef_video_encoder_config_set_h264_level(
    c: *mut EfVideoEncoderConfig,
    level: u32,
) -> c_int {
    unsafe {
        with_config(c, |c| {
            c.level = match level {
                0 => None,
                n => Some(
                    level_from(n).ok_or_else(|| invalid(&format!("{n} is not an H.264 level")))?,
                ),
            };
            Ok(())
        })
    }
}

/// Sets the JPEG quality of MJPEG pictures, 1 (smallest) to 100 (best);
/// 0 uses the default, 85.
///
/// @return 0, or `EINVAL` above 100.
///
/// # Safety
/// `c` must be a live configuration.
#[no_mangle]
pub unsafe extern "C" fn ef_video_encoder_config_set_quality(
    c: *mut EfVideoEncoderConfig,
    quality: u32,
) -> c_int {
    unsafe {
        with_config(c, |c| {
            c.quality = match quality {
                0 => None,
                q @ 1..=100 => Some(q as u8),
                q => return Err(invalid(&format!("JPEG quality {q} is not within 1 to 100"))),
            };
            Ok(())
        })
    }
}

/// One encoded access unit.
#[repr(C)]
pub struct EfVideoPacket {
    /// The bytes: Annex B for H.264, a JPEG file for MJPEG.
    pub data: *const u8,
    /// Bytes at `data`.
    pub len: usize,
    /// The `pts` the source frame was passed with.
    pub pts: u64,
    /// The access unit is a key frame (an IDR picture).
    pub keyframe: bool,
}

/// What runs an open encoder, and how its sources reach it.
#[repr(C)]
pub struct EfVideoEncoderInfo {
    /// The engine, an `ef_codec_engine`.
    pub engine: u32,
    /// How a whole source of the configured size, in the memory the engine
    /// prefers (DMA-BUF; a D3D11 texture on Windows), reaches the encoder,
    /// an `ef_codec_transfer`. Crops and host-memory sources may take a
    /// copy where this says none. A Media Foundation hardware encoder
    /// decides on its first frame whether it reads sources in place, and
    /// reports `EF_CODEC_TRANSFER_GPU_COPY` until then.
    pub transfer: u32,
    /// The encoder is dedicated hardware rather than software the platform
    /// provides.
    pub hardware: bool,
    /// The device within the engine: a device node, or a transform's name.
    pub device: *const c_char,
}

/// An opaque video encoder.
pub struct EfVideoEncoder {
    inner: VideoEncoder,
    ready: VecDeque<EncodedFrame>,
    /// The access unit [`ef_video_encoder_next_packet`] last returned.
    current: Option<EncodedFrame>,
    packet: EfVideoPacket,
    device: CString,
    info: EfVideoEncoderInfo,
}

/// Opens an encoder for the configuration, on the first engine to try that
/// runs the codec.
///
/// Platforms: Linux (V4L2 memory-to-memory encoders), Windows (Media
/// Foundation hardware encoder transforms). Elsewhere there is no engine
/// and this fails with `ENODEV`.
///
/// On Linux the first V4L2 device that produces the codec and accepts the
/// input format is used (`EDGEFIRST_CODEC_V4L2_ENCODER` names one device
/// node). On Windows the first hardware encoder transform that takes the
/// input format is used (`EDGEFIRST_CODEC_MF_ENCODER` keeps only the
/// transforms whose name contains its value). There is no software video
/// encoder.
///
/// @return the encoder, freed with [`ef_video_encoder_free`]; `NULL` with
///         `errno` `EINVAL` for an invalid configuration or one the device
///         rejects, `ENOTSUP` when no engine here encodes the codec or takes
///         the input format, or `ENODEV` when there is no encoder.
///
/// # Safety
/// `c` must be a live configuration; it is not kept.
#[no_mangle]
pub unsafe extern "C" fn ef_video_encoder_new(
    c: *const EfVideoEncoderConfig,
) -> *mut EfVideoEncoder {
    guard_new(|| {
        let c = unsafe { c.as_ref() }.ok_or_else(|| invalid("the configuration is NULL"))?;
        let inner = VideoEncoder::new(c.inner.clone()).map_err(|e| fail(&e))?;
        Ok(EfVideoEncoder {
            inner,
            ready: VecDeque::new(),
            current: None,
            packet: EfVideoPacket {
                data: std::ptr::null(),
                len: 0,
                pts: 0,
                keyframe: false,
            },
            device: CString::default(),
            info: EfVideoEncoderInfo {
                engine: 0,
                transfer: 0,
                hardware: false,
                device: std::ptr::null(),
            },
        })
    })
}

/// Frees an encoder, dropping access units not yet pulled. Freeing `NULL` is
/// a no-op.
///
/// # Safety
/// `e` must be `NULL` or have come from [`ef_video_encoder_new`].
#[no_mangle]
pub unsafe extern "C" fn ef_video_encoder_free(e: *mut EfVideoEncoder) {
    if !e.is_null() {
        drop(unsafe { Box::from_raw(e) });
    }
}

/// Encodes `src`, queueing any access unit that is ready.
///
/// `src` must have the configured pixel format and, without `crop`, the
/// configured size. `crop`, when not `NULL`, is the part of `src` to
/// encode, exactly the configured size. DMA-BUF and D3D11 sources are read
/// in place: the call returns once the device has finished reading `src`,
/// so the caller may reuse it afterwards. Every source of one stream must
/// have the same row stride. `pts` is returned unchanged on the access
/// unit; `flags` is 0 or `EF_VIDEO_ENCODE_KEYFRAME`.
///
/// Platforms: Linux, Windows.
///
/// @return 0; `EINVAL` for a source that does not match the configuration;
///         `ETIMEDOUT` when the device does not finish in time; otherwise an
///         errno, with the reason in `ef_codec_last_error`.
///
/// # Safety
/// `e` must be a live encoder, `src` a live tensor handle, and `crop`
/// `NULL` or readable.
#[no_mangle]
pub unsafe extern "C" fn ef_video_encoder_encode(
    e: *mut EfVideoEncoder,
    src: *const EfTensor,
    pts: u64,
    flags: u32,
    crop: *const EfCodecRegion,
) -> c_int {
    guard(|| {
        let Some(e) = (unsafe { e.as_mut() }) else {
            return invalid("the encoder is NULL");
        };
        if src.is_null() {
            return invalid("the source tensor is NULL");
        }
        if flags & !EF_VIDEO_ENCODE_KEYFRAME != 0 {
            return invalid(&format!("unknown encode flags {flags:#x}"));
        }
        let opts = FrameOptions {
            pts,
            crop: unsafe { crop.as_ref() }.map(|r| {
                Region::new(
                    r.x as usize,
                    r.y as usize,
                    r.width as usize,
                    r.height as usize,
                )
            }),
            keyframe: flags & EF_VIDEO_ENCODE_KEYFRAME != 0,
        };
        let EfVideoEncoder { inner, ready, .. } = e;
        match unsafe { TensorDyn::with_raw(src as *mut EfTensor, |t| inner.encode(t, &opts)) } {
            Ok(au) => {
                ready.extend(au);
                0
            }
            Err(err) => fail(&err),
        }
    })
}

/// Drains the encoder, queueing every access unit not yet queued. The
/// encoder accepts new frames afterwards.
///
/// @return 0, or an errno.
///
/// # Safety
/// `e` must be a live encoder.
#[no_mangle]
pub unsafe extern "C" fn ef_video_encoder_flush(e: *mut EfVideoEncoder) -> c_int {
    guard(|| {
        let Some(e) = (unsafe { e.as_mut() }) else {
            return invalid("the encoder is NULL");
        };
        match e.inner.flush() {
            Ok(aus) => {
                e.ready.extend(aus);
                0
            }
            Err(err) => fail(&err),
        }
    })
}

/// The next queued access unit, in decode order, or `NULL` when none is
/// queued. It belongs to the encoder and stays valid until the next call of
/// this function or until the encoder is freed.
///
/// # Safety
/// `e` must be `NULL` or a live encoder.
#[no_mangle]
pub unsafe extern "C" fn ef_video_encoder_next_packet(
    e: *mut EfVideoEncoder,
) -> *const EfVideoPacket {
    let Some(e) = (unsafe { e.as_mut() }) else {
        return std::ptr::null();
    };
    e.current = e.ready.pop_front();
    match &e.current {
        Some(au) => {
            e.packet = EfVideoPacket {
                data: au.data.as_ptr(),
                len: au.data.len(),
                pts: au.pts,
                keyframe: au.keyframe,
            };
            &e.packet
        }
        None => std::ptr::null(),
    }
}

/// The engine and device that encode, and how sources reach them, valid
/// until the next call of this function or until the encoder is freed;
/// `NULL` for a `NULL` encoder.
///
/// # Safety
/// `e` must be `NULL` or a live encoder.
#[no_mangle]
pub unsafe extern "C" fn ef_video_encoder_get_info(
    e: *mut EfVideoEncoder,
) -> *const EfVideoEncoderInfo {
    let Some(e) = (unsafe { e.as_mut() }) else {
        return std::ptr::null();
    };
    let info = e.inner.info();
    e.device = c_string(&info.device);
    e.info = EfVideoEncoderInfo {
        engine: engine_code(info.engine),
        transfer: transfer_code(info.transfer),
        hardware: info.hardware,
        device: e.device.as_ptr(),
    };
    &e.info
}

/// Geometry and format of a stream's decoded pictures.
#[repr(C)]
pub struct EfVideoStreamInfo {
    /// Width of the visible picture in pixels.
    pub width: u32,
    /// Height of the visible picture in pixels.
    pub height: u32,
    /// The visible picture within the coded one.
    pub crop: EfCodecRegion,
    /// Width of the coded picture.
    pub coded_width: u32,
    /// Height of the coded picture.
    pub coded_height: u32,
    /// Pixel format of the decoded frames (`"NV12"`).
    pub format: *const c_char,
    /// Colorimetry the device reports, packed as `ef_tensor_colorimetry`
    /// returns it; 0 when it reports none.
    pub colorimetry: u32,
    /// Buffers in the output pool.
    pub buffers: u32,
    /// Whose memory the pictures are decoded into, an `ef_video_memory`.
    pub memory: u32,
    /// The engine that decodes, an `ef_codec_engine`.
    pub engine: u32,
    /// How each picture reaches its pool tensor, an `ef_codec_transfer`:
    /// written in place (V4L2), or copied out of the decoder's own surfaces
    /// on the GPU (Media Foundation with D3D11) or through CPU memory.
    pub transfer: u32,
}

/// An [`EfVideoStreamInfo`] and the string it points into.
struct StreamInfoC {
    info: EfVideoStreamInfo,
    _format: CString,
}

impl StreamInfoC {
    fn from(s: &StreamInfo) -> Box<Self> {
        let c = |v: usize| u32::try_from(v).unwrap_or(u32::MAX);
        let format = c_string(s.format.as_str());
        Box::new(Self {
            info: EfVideoStreamInfo {
                width: c(s.width),
                height: c(s.height),
                crop: s.crop.into(),
                coded_width: c(s.coded_width),
                coded_height: c(s.coded_height),
                format: format.as_ptr(),
                colorimetry: s.colorimetry.map_or(0, |c| c.pack()),
                buffers: c(s.buffers),
                memory: match s.memory {
                    OutputMemory::Imported => EfVideoMemory::Imported,
                    OutputMemory::Exported => EfVideoMemory::Exported,
                } as u32,
                engine: engine_code(s.engine),
                transfer: transfer_code(s.transfer),
            },
            _format: format,
        })
    }
}

/// An opaque video decoder.
pub struct EfVideoDecoder {
    inner: VideoDecoder,
    events: VecDeque<DecodeEvent>,
    /// The stream info of the last `EF_VIDEO_EVENT_RECONFIGURED` returned.
    info: Option<Box<StreamInfoC>>,
}

/// An opaque decoded frame. It holds its pool buffer until it is freed.
pub struct EfVideoFrame {
    inner: DecodedFrame,
}

fn open_decoder(config: DecoderConfig) -> Result<EfVideoDecoder, c_int> {
    Ok(EfVideoDecoder {
        inner: VideoDecoder::new(config).map_err(|e| fail(&e))?,
        events: VecDeque::new(),
        info: None,
    })
}

/// Opens a decoder for `codec` (an `ef_codec`) on `engine` (an
/// `ef_codec_engine`), which allocates its own frames: DMA-BUF tensors on
/// Linux, D3D11 textures on Windows (host memory where the process has no
/// D3D11 device), allocated again when the stream changes size.
///
/// `pool_count` is the frames the caller may hold at once, 4 if 0; the
/// device's own reference pictures and one buffer to decode into are added
/// to it.
///
/// Platforms: Linux (V4L2 memory-to-memory decoders), Windows (Media
/// Foundation: Microsoft's H.264 decoder with DXVA, and MJPEG decoders).
/// Elsewhere there is no engine and this fails with `ENODEV`.
///
/// @return the decoder, freed with [`ef_video_decoder_free`]; `NULL` with
///         `errno` `ENOTSUP` when no engine here decodes the codec, `ENODEV`
///         when there is no decoder, or `EINVAL`.
#[no_mangle]
pub extern "C" fn ef_video_decoder_new(
    codec: u32,
    engine: u32,
    pool_count: usize,
) -> *mut EfVideoDecoder {
    guard_new(|| {
        let config = DecoderConfig::new(codec_arg(codec)?)
            .with_engine(engine_arg(engine)?)
            .with_output(OutputPool::Allocate {
                count: if pool_count == 0 { 4 } else { pool_count },
            });
        open_decoder(config)
    })
}

/// Opens a decoder that decodes into the caller's tensors, one buffer each.
/// Each must be a different buffer: a pool that holds one buffer twice, as
/// the same handle or two handles over it, is `EINVAL`.
///
/// On success the decoder takes over the caller's reference to each tensor:
/// the caller must not free them. A caller that keeps using one takes its
/// own reference first with `ef_tensor_retain`; while that reference is
/// held, the decoder's CPU writes into the tensor (software decoders) are
/// refused. On failure the caller keeps its references.
///
/// On Windows pictures are copied into the pool, and each tensor must be an
/// NV12 image of exactly the stream's visible size; such a pool cannot
/// follow a change of size. On Linux the device imports the tensors'
/// DMA-BUFs: there must be at least one more than the device keeps for
/// reference pictures, each large enough for the coded picture at the row
/// stride the device asks for, or decoding fails with `EINVAL` once the
/// stream's size is known.
///
/// Platforms: Linux, Windows.
///
/// @return as [`ef_video_decoder_new`].
///
/// # Safety
/// `pool` must point to `count` live tensor handles.
#[no_mangle]
pub unsafe extern "C" fn ef_video_decoder_new_with_pool(
    codec: u32,
    engine: u32,
    pool: *const *mut EfTensor,
    count: usize,
) -> *mut EfVideoDecoder {
    guard_new(|| {
        let codec = codec_arg(codec)?;
        let engine = engine_arg(engine)?;
        if pool.is_null() || count == 0 {
            return Err(invalid("the pool is empty"));
        }
        let raws = unsafe { std::slice::from_raw_parts(pool, count) };
        if raws.iter().any(|t| t.is_null()) {
            return Err(invalid("a pool tensor is NULL"));
        }
        // The decoder's references are taken here and the caller's released
        // only on success, so a failure leaves the caller's intact.
        let mut tensors = Vec::with_capacity(count);
        for &raw in raws {
            if unsafe { edgefirst_tensor_ffi::ef_tensor_retain(raw) } != 0 {
                return Err(invalid("a pool tensor could not be retained"));
            }
            tensors.push(unsafe { TensorDyn::from_raw(raw) });
        }
        let config = DecoderConfig::new(codec)
            .with_engine(engine)
            .with_output(OutputPool::Provided(tensors));
        let decoder = open_decoder(config)?;
        for &raw in raws {
            unsafe { edgefirst_tensor_ffi::ef_tensor_free(raw) };
        }
        Ok(decoder)
    })
}

/// Frees a decoder, dropping events not yet pulled. Frames already pulled
/// stay valid until they are freed. Freeing `NULL` is a no-op.
///
/// # Safety
/// `d` must be `NULL` or have come from [`ef_video_decoder_new`] or
/// [`ef_video_decoder_new_with_pool`].
#[no_mangle]
pub unsafe extern "C" fn ef_video_decoder_free(d: *mut EfVideoDecoder) {
    if !d.is_null() {
        drop(unsafe { Box::from_raw(d) });
    }
}

/// Decodes one access unit, queueing every frame and format change that is
/// ready.
///
/// Access units go in decode order: for H.264 an Annex B access unit (see
/// `edgefirst_codec::video::access_units` for splitting a byte stream), for
/// MJPEG one JPEG picture. `pts` is returned unchanged on the frame.
///
/// Platforms: Linux, Windows.
///
/// @return 0; `EAGAIN` when the caller holds every frame of the pool (free
///         some, then decode the same access unit again); `EBADMSG` for data
///         the decoder cannot parse; `EINVAL` for a provided pool that does
///         not fit the stream; `ETIMEDOUT` when the device does not finish
///         in time; otherwise an errno, with the reason in
///         `ef_codec_last_error`.
///
/// # Safety
/// `d` must be a live decoder and `data` readable for `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn ef_video_decoder_decode(
    d: *mut EfVideoDecoder,
    data: *const u8,
    len: usize,
    pts: u64,
) -> c_int {
    guard(|| {
        let Some(d) = (unsafe { d.as_mut() }) else {
            return invalid("the decoder is NULL");
        };
        if data.is_null() || len == 0 {
            return invalid("the access unit is empty");
        }
        let au = unsafe { std::slice::from_raw_parts(data, len) };
        match d.inner.decode(au, pts) {
            Ok(events) => {
                d.events.extend(events);
                0
            }
            Err(err) => fail(&err),
        }
    })
}

/// Drains the decoder at the end of the stream, queueing every frame not yet
/// queued.
///
/// @return 0, or an errno.
///
/// # Safety
/// `d` must be a live decoder.
#[no_mangle]
pub unsafe extern "C" fn ef_video_decoder_flush(d: *mut EfVideoDecoder) -> c_int {
    guard(|| {
        let Some(d) = (unsafe { d.as_mut() }) else {
            return invalid("the decoder is NULL");
        };
        match d.inner.flush() {
            Ok(events) => {
                d.events.extend(events);
                0
            }
            Err(err) => fail(&err),
        }
    })
}

/// Pulls the next queued event into `*kind`, an `ef_video_event`.
///
/// For `EF_VIDEO_EVENT_FRAME`, `*frame` receives the frame, which the caller
/// frees with [`ef_video_frame_free`] to give its buffer back to the pool;
/// otherwise `*frame` is `NULL`. For `EF_VIDEO_EVENT_RECONFIGURED`,
/// [`ef_video_decoder_stream_info`] now describes the frames that follow.
/// `EF_VIDEO_EVENT_NONE` means the queue is empty.
///
/// @return 0, or `EINVAL`.
///
/// # Safety
/// `d` must be a live decoder; `kind` and `frame` writable.
#[no_mangle]
pub unsafe extern "C" fn ef_video_decoder_next_event(
    d: *mut EfVideoDecoder,
    kind: *mut u32,
    frame: *mut *mut EfVideoFrame,
) -> c_int {
    guard(|| {
        if kind.is_null() || frame.is_null() {
            return invalid("kind and frame must not be NULL");
        }
        unsafe {
            *kind = EfVideoEvent::None as u32;
            *frame = std::ptr::null_mut();
        }
        let Some(d) = (unsafe { d.as_mut() }) else {
            return invalid("the decoder is NULL");
        };
        match d.events.pop_front() {
            Some(DecodeEvent::Frame(f)) => unsafe {
                *kind = EfVideoEvent::Frame as u32;
                *frame = Box::into_raw(Box::new(EfVideoFrame { inner: f }));
            },
            Some(DecodeEvent::Reconfigured(info)) => {
                d.info = Some(StreamInfoC::from(&info));
                unsafe { *kind = EfVideoEvent::Reconfigured as u32 };
            }
            None => {}
        }
        0
    })
}

/// The stream's geometry and format as of the last
/// `EF_VIDEO_EVENT_RECONFIGURED` pulled, valid until the next one is pulled
/// or the decoder is freed; `NULL` before the first.
///
/// # Safety
/// `d` must be `NULL` or a live decoder.
#[no_mangle]
pub unsafe extern "C" fn ef_video_decoder_stream_info(
    d: *const EfVideoDecoder,
) -> *const EfVideoStreamInfo {
    unsafe { d.as_ref() }
        .and_then(|d| d.info.as_deref())
        .map_or(std::ptr::null(), |i| &i.info)
}

/// The decoded picture: an image tensor of the visible size, `NULL` for a
/// `NULL` frame.
///
/// The handle belongs to the frame and is valid until the frame is freed;
/// do not free it, or change its shape or format. Pass it to any EdgeFirst
/// library as a source. A caller that keeps it past the frame takes its own
/// reference with `ef_tensor_retain`, but the pool may decode a later
/// picture into the same memory once the frame is freed.
///
/// # Safety
/// `f` must be `NULL` or a live frame.
#[no_mangle]
pub unsafe extern "C" fn ef_video_frame_tensor(f: *const EfVideoFrame) -> *mut EfTensor {
    unsafe { f.as_ref() }.map_or(std::ptr::null_mut(), |f| f.inner.tensor().as_raw())
}

/// The `pts` passed with the frame's access unit; 0 for `NULL`.
///
/// # Safety
/// `f` must be `NULL` or a live frame.
#[no_mangle]
pub unsafe extern "C" fn ef_video_frame_pts(f: *const EfVideoFrame) -> u64 {
    unsafe { f.as_ref() }.map_or(0, |f| f.inner.pts())
}

/// Whether the device reported an error decoding this picture, so it may be
/// partly wrong (a missing reference, a damaged slice).
///
/// # Safety
/// `f` must be `NULL` or a live frame.
#[no_mangle]
pub unsafe extern "C" fn ef_video_frame_corrupted(f: *const EfVideoFrame) -> bool {
    unsafe { f.as_ref() }.is_some_and(|f| f.inner.corrupted())
}

/// Frees a frame, giving its buffer back to the decoder's pool. Freeing
/// `NULL` is a no-op.
///
/// # Safety
/// `f` must be `NULL` or have come from [`ef_video_decoder_next_event`].
#[no_mangle]
pub unsafe extern "C" fn ef_video_frame_free(f: *mut EfVideoFrame) {
    if !f.is_null() {
        drop(unsafe { Box::from_raw(f) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{EfCodec, EfCodecEngine};

    /// The one struct a caller fills: four `uint32_t`, as the header says.
    #[test]
    fn a_region_is_four_u32() {
        assert_eq!(std::mem::size_of::<EfCodecRegion>(), 16);
        assert_eq!(std::mem::align_of::<EfCodecRegion>(), 4);
    }

    #[test]
    fn every_h264_level_has_a_code() {
        let codes = [
            9, 10, 11, 12, 13, 20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52, 60, 61, 62,
        ];
        let mut levels: Vec<H264Level> = codes.iter().map(|&c| level_from(c).unwrap()).collect();
        levels.dedup();
        assert_eq!(levels.len(), codes.len());
        assert!(level_from(14).is_none());
        assert!(level_from(1).is_none());
    }

    #[test]
    fn the_configuration_setters_check_their_values() {
        unsafe {
            let c =
                ef_video_encoder_config_new(EfCodec::H264 as u32, 64, 64, c"NV12".as_ptr(), 30.0);
            assert!(!c.is_null());
            assert_eq!(ef_video_encoder_config_set_engine(c, 9), libc::EINVAL);
            assert_eq!(
                ef_video_encoder_config_set_engine(c, EfCodecEngine::V4l2 as u32),
                0
            );
            assert_eq!(ef_video_encoder_config_set_bitrate(c, 2_000_000), 0);
            assert_eq!(ef_video_encoder_config_set_gop(c, 30), 0);
            assert_eq!(ef_video_encoder_config_set_repeat_headers(c, false), 0);
            assert_eq!(ef_video_encoder_config_set_h264_profile(c, 3), 0);
            assert_eq!(ef_video_encoder_config_set_h264_profile(c, 5), libc::EINVAL);
            assert_eq!(ef_video_encoder_config_set_h264_level(c, 41), 0);
            assert_eq!(ef_video_encoder_config_set_h264_level(c, 43), libc::EINVAL);
            assert_eq!(ef_video_encoder_config_set_quality(c, 101), libc::EINVAL);
            assert_eq!(ef_video_encoder_config_set_quality(c, 70), 0);
            let inner = &(*c).inner;
            assert_eq!(inner.engine, edgefirst_codec::Engine::V4l2);
            assert_eq!(inner.bitrate, Bitrate::Bps(2_000_000));
            assert_eq!(inner.gop, Some(30));
            assert!(!inner.repeat_headers);
            assert_eq!(inner.profile, Some(H264Profile::Main));
            assert_eq!(inner.level, Some(H264Level::L4_1));
            assert_eq!(inner.quality, Some(70));
            ef_video_encoder_config_free(c);

            assert!(ef_video_encoder_config_new(2, 64, 64, c"NV99".as_ptr(), 30.0).is_null());
            assert_eq!(errno::errno().0, libc::EINVAL);
            assert!(ef_video_encoder_config_new(2, 64, 64, std::ptr::null(), 30.0).is_null());
        }
    }

    /// A codec no engine runs is refused at open, with the errno a caller
    /// can branch on.
    #[test]
    fn a_codec_no_engine_runs_is_enotsup_or_enodev() {
        let d = ef_video_decoder_new(EfCodec::Av1 as u32, EfCodecEngine::Auto as u32, 0);
        assert!(d.is_null());
        assert!(matches!(errno::errno().0, libc::ENOTSUP | libc::ENODEV));
        assert!(!crate::error::ef_codec_last_error().is_null());
        unsafe {
            let c =
                ef_video_encoder_config_new(EfCodec::Vp9 as u32, 64, 64, c"NV12".as_ptr(), 30.0);
            assert!(ef_video_encoder_new(c).is_null());
            assert!(matches!(errno::errno().0, libc::ENOTSUP | libc::ENODEV));
            ef_video_encoder_config_free(c);
        }
    }

    #[test]
    fn null_handles_are_errors_not_crashes() {
        unsafe {
            let mut kind = 7;
            let mut frame = std::ptr::NonNull::dangling().as_ptr();
            assert_eq!(
                ef_video_decoder_next_event(std::ptr::null_mut(), &mut kind, &mut frame),
                libc::EINVAL
            );
            assert_eq!(kind, EfVideoEvent::None as u32);
            assert!(frame.is_null());
            assert_eq!(
                ef_video_decoder_decode(std::ptr::null_mut(), [0u8].as_ptr(), 1, 0),
                libc::EINVAL
            );
            assert_eq!(ef_video_decoder_flush(std::ptr::null_mut()), libc::EINVAL);
            assert!(ef_video_decoder_stream_info(std::ptr::null()).is_null());
            assert!(ef_video_encoder_next_packet(std::ptr::null_mut()).is_null());
            assert!(ef_video_encoder_get_info(std::ptr::null_mut()).is_null());
            assert_eq!(
                ef_video_encoder_encode(
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    0,
                    0,
                    std::ptr::null()
                ),
                libc::EINVAL
            );
            assert!(ef_video_frame_tensor(std::ptr::null()).is_null());
            assert_eq!(ef_video_frame_pts(std::ptr::null()), 0);
            assert!(!ef_video_frame_corrupted(std::ptr::null()));
            ef_video_frame_free(std::ptr::null_mut());
            ef_video_decoder_free(std::ptr::null_mut());
            ef_video_encoder_free(std::ptr::null_mut());
            assert!(ef_video_decoder_new_with_pool(2, 0, std::ptr::null(), 0).is_null());
            assert_eq!(errno::errno().0, libc::EINVAL);
        }
    }

    #[test]
    fn a_pool_holding_one_tensor_twice_is_refused_and_keeps_its_references() {
        use edgefirst_tensor::{CpuAccess, DType, PixelFormat, TensorMemory};
        let t = TensorDyn::image(
            64,
            64,
            PixelFormat::Nv12,
            DType::U8,
            Some(TensorMemory::Mem),
            CpuAccess::ReadWrite,
        )
        .unwrap()
        .into_raw();
        unsafe {
            assert_eq!(edgefirst_tensor_ffi::ef_tensor_retain(t), 0);
            let pool = [t, t];
            let d = ef_video_decoder_new_with_pool(
                EfCodec::H264 as u32,
                EfCodecEngine::Auto as u32,
                pool.as_ptr(),
                pool.len(),
            );
            assert!(d.is_null());
            assert_eq!(errno::errno().0, libc::EINVAL);
            let msg = std::ffi::CStr::from_ptr(crate::error::ef_codec_last_error());
            assert!(msg.to_str().unwrap().contains("same buffer"), "{msg:?}");
            // The caller still holds exactly its two references: a CPU write
            // map is refused while the handle is shared, and succeeds once one
            // is released.
            assert!(!TensorDyn::with_raw(t, |t| t
                .map_bytes(CpuAccess::Write)
                .is_ok()));
            edgefirst_tensor_ffi::ef_tensor_free(t);
            assert!(TensorDyn::with_raw(t, |t| t
                .map_bytes(CpuAccess::Write)
                .is_ok()));
            edgefirst_tensor_ffi::ef_tensor_free(t);
        }
    }
}
