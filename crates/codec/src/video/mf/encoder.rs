// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! H.264 encoding with a Media Foundation encoder transform.
//!
//! A hardware encoder that takes D3D11 textures is given a DXGI device
//! manager on HAL's device, and always reads a texture. A native NV12
//! texture source of the configured size is handed to it in place when the
//! encoder returns each frame's access unit at once: `encode` then waits for
//! it, so the encoder has read the source by the time the call returns.
//! Otherwise the source is copied into one of a ring of scratch textures --
//! on the GPU from a texture (a crop, or a source for an encoder that holds
//! frames), on the CPU from a host tensor -- which stays the encoder's until
//! its access unit comes back. The first frame takes the copy and measures
//! which kind of encoder this is.
//!
//! An encoder without D3D11 support (Microsoft's software encoder) gets each
//! source copied, tightly packed, into a system-memory sample instead.
//!
//! Sample times are the frame's sequence number in frame periods, and the
//! caller's PTS is looked up again by the time of the access unit that comes
//! back. B-frames are disabled, so access units come back in the order their
//! sources went in.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use edgefirst_tensor::{CpuAccess, DType, PixelFormat, TensorDyn, TensorMemory};
use windows::core::{Interface, GUID};
use windows::Win32::Graphics::Direct3D11::{ID3D11DeviceContext, D3D11_BOX};
use windows::Win32::Media::MediaFoundation::IMFDXGIDeviceManager;
use windows::Win32::Media::MediaFoundation::{
    eAVEncCommonRateControlMode_CBR, CODECAPI_AVEncCommonMeanBitRate,
    CODECAPI_AVEncCommonRateControlMode, CODECAPI_AVEncMPVDefaultBPictureCount,
    CODECAPI_AVEncMPVGOPSize, CODECAPI_AVEncVideoForceKeyFrame, CODECAPI_AVLowLatencyMode,
    ICodecAPI, IMFAttributes, IMFMediaType, IMFSample, MFMediaType_Video,
    MFSampleExtension_CleanPoint, MFVideoFormat_H264, MFVideoFormat_NV12, MFVideoFormat_YUY2,
    MFVideoInterlace_Progressive, MFT_CATEGORY_VIDEO_ENCODER, MF_MT_AVG_BITRATE,
    MF_MT_DEFAULT_STRIDE, MF_MT_FRAME_RATE, MF_MT_FRAME_SIZE, MF_MT_INTERLACE_MODE,
    MF_MT_MAJOR_TYPE, MF_MT_MPEG2_LEVEL, MF_MT_MPEG2_PROFILE, MF_MT_MPEG_SEQUENCE_HEADER,
    MF_MT_PIXEL_ASPECT_RATIO, MF_MT_SUBTYPE,
};
use windows::Win32::System::Variant::VARIANT;

use super::copy::{self, Plane};
use super::{
    com, error, frame_duration, frame_rate_ratio, hal_device, is_native_nv12, set_pair,
    tensor_texture, Output, Platform, Transform,
};
use crate::video::h264::{self, AccessUnit};
use crate::video::{
    chroma_subsampling, EncodedFrame, EncoderBackend, EncoderConfig, FrameOptions, H264Level,
    H264Profile,
};
use crate::{CodecError, Result};

/// Environment variable keeping only the encoders whose friendly name
/// contains its value (or `hardware`, or `software`).
const ENV_ENCODER: &str = "EDGEFIRST_CODEC_MF_ENCODER";
/// How long the encoder may take to accept one source frame.
const INPUT_TIMEOUT: Duration = Duration::from_secs(2);
/// How long an asynchronous encoder may take, once a source is accepted,
/// to return an access unit before `encode` returns without one.
const OUTPUT_GRACE: Duration = Duration::from_millis(100);
/// How long a drain may take.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(2);
/// How long a D3D11 encoder may take to return a frame's access unit and
/// still read its sources in place.
const IN_PLACE_WAIT: Duration = Duration::from_millis(50);

/// The Media Foundation subtype that carries `format`.
fn input_subtype(format: PixelFormat) -> Option<GUID> {
    match format {
        PixelFormat::Nv12 => Some(MFVideoFormat_NV12),
        PixelFormat::Yuyv => Some(MFVideoFormat_YUY2),
        _ => None,
    }
}

/// `eAVEncH264VProfile` of `profile`.
fn profile_value(profile: H264Profile) -> u32 {
    match profile {
        H264Profile::Baseline => 66,
        H264Profile::ConstrainedBaseline => 256,
        H264Profile::Main => 77,
        H264Profile::High => 100,
    }
}

/// `eAVEncH264VLevel` of `level`: its `level_idc`, with level 1b as 11.
fn level_value(level: H264Level) -> u32 {
    use H264Level as L;
    match level {
        L::L1_0 => 10,
        L::L1b | L::L1_1 => 11,
        L::L1_2 => 12,
        L::L1_3 => 13,
        L::L2_0 => 20,
        L::L2_1 => 21,
        L::L2_2 => 22,
        L::L3_0 => 30,
        L::L3_1 => 31,
        L::L3_2 => 32,
        L::L4_0 => 40,
        L::L4_1 => 41,
        L::L4_2 => 42,
        L::L5_0 => 50,
        L::L5_1 => 51,
        L::L5_2 => 52,
        L::L6_0 => 60,
        L::L6_1 => 61,
        L::L6_2 => 62,
    }
}

/// The profile and level of each output type to try, in order: the
/// requested pair, then, for whichever of the two the caller left unset,
/// the encoder's own default.
fn output_attempts(config: &EncoderConfig) -> Vec<(Option<H264Profile>, Option<H264Level>)> {
    let profile = config.target_profile();
    let level = config.target_level();
    let strict_level = config.level.is_some();
    let mut attempts = vec![(Some(profile), level)];
    if !strict_level && level.is_some() {
        attempts.push((Some(profile), None));
    }
    if config.profile.is_none() {
        attempts.push((None, if strict_level { level } else { None }));
    }
    attempts
}

/// Bytes per row of a tightly packed `format` picture `width` wide.
fn tight_pitch(format: PixelFormat, width: usize) -> usize {
    match format {
        PixelFormat::Yuyv => width * 2,
        _ => width,
    }
}

/// What an encoder reading D3D11 textures needs.
struct TextureInput {
    context: ID3D11DeviceContext,
    /// Native NV12 textures of the configured size for sources that are not
    /// read in place, each with the sample time of the frame it holds while
    /// that frame's access unit has not come back. Grown when every one is
    /// still the encoder's.
    ring: Vec<(TensorDyn, Option<i64>)>,
    /// Whether whole texture sources are read in place: `None` until the
    /// first frame has shown whether the encoder returns access units at
    /// once.
    in_place: Option<bool>,
}

pub(crate) struct MfEncoder {
    transform: Transform,
    codec_api: Option<ICodecAPI>,
    /// `Some` when the encoder reads D3D11 textures.
    textures: Option<TextureInput>,
    /// Kept for the transform, which holds its own reference.
    _manager: Option<IMFDXGIDeviceManager>,
    config: EncoderConfig,
    device: String,
    /// One frame period in 100 ns units.
    duration: i64,
    /// Planes of the packed input sample.
    planes: Vec<Plane>,
    frame_bytes: usize,
    next_seq: i64,
    /// `(sample time, pts)` of sources whose access unit has not come back.
    in_flight: VecDeque<(i64, u64)>,
    pending: VecDeque<EncodedFrame>,
    /// NAL units that precede a picture, delivered in a sample of their own,
    /// for the next access unit.
    headers: Vec<u8>,
    /// The stream's latest SPS and PPS, for key frames sent without them.
    parameter_sets: Vec<u8>,
    /// The next frame starts a new sequence: an IDR with SPS and PPS.
    restart: bool,
    /// Access units queued since the sequence started.
    queued: u64,
    /// Last, so Media Foundation stays started until the transform is
    /// released.
    platform: Platform,
}

// SAFETY: the transform and its codec interface are free-threaded Media
// Foundation objects, used through `&mut self` only, so from one thread at a
// time.
unsafe impl Send for MfEncoder {}

impl MfEncoder {
    /// Opens the first encoder transform that encodes H.264 from
    /// `config.input_format` and takes the configuration.
    pub fn open(config: EncoderConfig) -> Result<Self> {
        let subtype = input_subtype(config.input_format)
            .ok_or(CodecError::UnsupportedFormat(config.input_format))?;
        let platform = Platform::start()?;
        let candidates = platform.transforms(
            MFT_CATEGORY_VIDEO_ENCODER,
            Some(subtype),
            MFVideoFormat_H264,
            true,
            ENV_ENCODER,
        )?;
        if candidates.is_empty() {
            let any = platform.transforms(
                MFT_CATEGORY_VIDEO_ENCODER,
                None,
                MFVideoFormat_H264,
                true,
                ENV_ENCODER,
            )?;
            if any.is_empty() {
                return Err(CodecError::NoDevice(match std::env::var(ENV_ENCODER) {
                    Ok(f) if !f.is_empty() => {
                        format!("no Media Foundation H.264 encoder matches {ENV_ENCODER}={f}")
                    }
                    _ => "no Media Foundation H.264 encoder".into(),
                }));
            }
            return Err(CodecError::UnsupportedFormat(config.input_format));
        }
        // Only NV12 has a texture an encoder reads in place.
        let manager = (subtype == MFVideoFormat_NV12)
            .then(|| platform.device_manager())
            .flatten();
        let mut rejected = None;
        for candidate in candidates {
            let name = candidate.name.clone();
            match Transform::activate(candidate)
                .and_then(|t| Self::configure(&platform, t, &config, subtype, manager.as_ref()))
            {
                Ok((transform, codec_api, d3d)) => {
                    let textures = if d3d {
                        // SAFETY: a live device; its immediate context lives
                        // as long as it does.
                        let context = unsafe { hal_device()?.GetImmediateContext() }
                            .map_err(|e| error(e, "ID3D11Device::GetImmediateContext"))?;
                        Some(TextureInput {
                            context,
                            ring: Vec::new(),
                            in_place: None,
                        })
                    } else {
                        None
                    };
                    let (w, h) = (config.width as usize, config.height as usize);
                    let pitch = tight_pitch(config.input_format, w);
                    let frame_bytes = copy::plane_shapes(config.input_format, w, h)
                        .unwrap_or_default()
                        .iter()
                        .map(|(len, rows)| len * rows)
                        .sum();
                    let device = format!(
                        "Media Foundation: {name} ({})",
                        match (transform.hardware(), textures.is_some()) {
                            (true, true) => "hardware, D3D11",
                            (true, false) => "hardware",
                            (false, _) => "software",
                        }
                    );
                    log::info!(
                        "h264 encoder {device} ({}x{} {}, {} bps)",
                        config.width,
                        config.height,
                        config.input_format,
                        config.target_bitrate()
                    );
                    let mut enc = Self {
                        transform,
                        codec_api,
                        textures,
                        _manager: manager.clone(),
                        duration: frame_duration(config.frame_rate),
                        planes: copy::contiguous(config.input_format, pitch, h),
                        frame_bytes,
                        platform,
                        config,
                        device,
                        next_seq: 0,
                        in_flight: VecDeque::new(),
                        pending: VecDeque::new(),
                        headers: Vec::new(),
                        parameter_sets: Vec::new(),
                        restart: false,
                        queued: 0,
                    };
                    enc.parameter_sets = enc.sequence_header();
                    return Ok(enc);
                }
                Err(e) => {
                    log::debug!("{name}: {e}");
                    // A refused profile or level says more than a later
                    // transform's failure to take the input format.
                    if rejected.is_none() || matches!(e, CodecError::InvalidConfig(_)) {
                        rejected = Some(e);
                    }
                }
            }
        }
        Err(rejected.expect("at least one candidate was tried"))
    }

    /// Applies `config` to `transform`, on HAL's device when it takes
    /// `manager`. Returns whether it reads D3D11 textures.
    fn configure(
        platform: &Platform,
        mut transform: Transform,
        config: &EncoderConfig,
        subtype: GUID,
        manager: Option<&IMFDXGIDeviceManager>,
    ) -> Result<(Transform, Option<ICodecAPI>, bool)> {
        // Before the media types, which the encoder checks against its
        // device.
        let d3d = manager.is_some_and(|m| transform.set_device_manager(m));
        let codec_api = transform.mft().cast::<ICodecAPI>().ok();
        // Rate control has to be chosen before the output type is set.
        if let Some(api) = &codec_api {
            apply_codec_api(api, config, transform.name());
        }

        let mut last = None;
        let attempts = output_attempts(config);
        let mut chosen = None;
        for &(profile, level) in &attempts {
            let t = output_type(platform, config, profile, level)?;
            match transform.set_output_type(&t) {
                Ok(()) => {
                    chosen = Some((profile, level));
                    break;
                }
                Err(e) => {
                    log::debug!("{}: output {profile:?} {level:?}: {e}", transform.name());
                    last = Some(e);
                }
            }
        }
        let Some((profile, level)) = chosen else {
            return Err(CodecError::InvalidConfig(format!(
                "{} rejects H.264 {:?} at level {:?}, {}x{} at {} fps, {} bps: {}",
                transform.name(),
                config.target_profile(),
                config.target_level(),
                config.width,
                config.height,
                config.frame_rate,
                config.target_bitrate(),
                last.map(|e| e.message()).unwrap_or_default()
            )));
        };
        if profile.is_none() || level.is_none() {
            log::debug!(
                "{}: encoding with its default {}",
                transform.name(),
                if profile.is_none() {
                    "profile and level"
                } else {
                    "level"
                }
            );
        }

        let input = video_type(platform, subtype, config)?;
        let attrs: IMFAttributes = input.cast().map_err(|e| error(e, "IMFAttributes"))?;
        let stride = tight_pitch(config.input_format, config.width as usize) as u32;
        // SAFETY: a live media type.
        unsafe { attrs.SetUINT32(&MF_MT_DEFAULT_STRIDE, stride) }
            .map_err(|e| error(e, "IMFAttributes::SetUINT32"))?;
        transform.set_input_type(&input).map_err(|e| {
            log::debug!("{}: input {}: {e}", transform.name(), config.input_format);
            CodecError::UnsupportedFormat(config.input_format)
        })?;
        if let Some(api) = &codec_api {
            apply_codec_api(api, config, transform.name());
        }
        let (w, h) = (config.width as usize, config.height as usize);
        transform.set_min_output_size((w * h * 3 / 2).max(1 << 16));
        transform.begin()?;
        Ok((transform, codec_api, d3d))
    }

    /// The SPS and PPS the output type carries, if the encoder sets them
    /// there.
    fn sequence_header(&self) -> Vec<u8> {
        let Ok(t) = self.transform.output_type() else {
            return Vec::new();
        };
        let Ok(attrs) = t.cast::<IMFAttributes>() else {
            return Vec::new();
        };
        // SAFETY: a live attribute store.
        let Ok(len) = (unsafe { attrs.GetBlobSize(&MF_MT_MPEG_SEQUENCE_HEADER) }) else {
            return Vec::new();
        };
        let mut blob = vec![0u8; len as usize];
        // SAFETY: as above, into a buffer of the blob's size.
        if unsafe { attrs.GetBlob(&MF_MT_MPEG_SEQUENCE_HEADER, &mut blob, None) }.is_err() {
            return Vec::new();
        }
        h264::parameter_sets(&blob)
    }

    fn check_source(&self, src: &TensorDyn, opts: &FrameOptions) -> Result<(usize, usize)> {
        let cfg = &self.config;
        let invalid = |what: String| Err(CodecError::InvalidInput(what));
        if src.format() != Some(cfg.input_format) {
            return invalid(format!(
                "source format {:?} is not the configured {}",
                src.format(),
                cfg.input_format
            ));
        }
        if src.dtype() != DType::U8 {
            return invalid(format!(
                "source elements are {:?}; {} is 8-bit",
                src.dtype(),
                cfg.input_format
            ));
        }
        let (sw, sh) = match (src.width(), src.height()) {
            (Some(w), Some(h)) => (w, h),
            _ => return invalid("source is not an image".into()),
        };
        if src.view_origin().is_some() && cfg.input_format == PixelFormat::Nv12 {
            return invalid(format!(
                "a view of a {} image cannot be encoded; crop with FrameOptions::crop",
                cfg.input_format
            ));
        }
        let (w, h) = (cfg.width as usize, cfg.height as usize);
        let Some(crop) = opts.crop else {
            if (sw, sh) != (w, h) {
                return invalid(format!("source is {sw}x{sh}, the encoder {w}x{h}"));
            }
            return Ok((0, 0));
        };
        if (crop.width, crop.height) != (w, h) {
            return invalid(format!(
                "crop is {}x{}, the encoder {w}x{h}",
                crop.width, crop.height
            ));
        }
        if crop.x + crop.width > sw || crop.y + crop.height > sh {
            return invalid(format!(
                "crop {}x{}+{}+{} is outside the {sw}x{sh} source",
                crop.width, crop.height, crop.x, crop.y
            ));
        }
        let (hsub, vsub) = chroma_subsampling(cfg.input_format);
        if !crop.x.is_multiple_of(hsub as usize) || !crop.y.is_multiple_of(vsub as usize) {
            return invalid(format!(
                "crop origin {},{} is not on a {} chroma sample",
                crop.x, crop.y, cfg.input_format
            ));
        }
        Ok((crop.x, crop.y))
    }

    /// A sample holding the region of `src` at `origin`, tightly packed.
    fn fill(&self, src: &TensorDyn, origin: (usize, usize)) -> Result<IMFSample> {
        let format = self.config.input_format;
        let size = (self.config.width as usize, self.config.height as usize);
        let stride = src
            .effective_row_stride()
            .ok_or_else(|| CodecError::InvalidInput("source has no row stride".into()))?;
        let (sample, buffer) = self.platform.sample(self.frame_bytes)?;
        let mut ptr = std::ptr::null_mut();
        // SAFETY: a live buffer; unlocked below.
        unsafe { buffer.Lock(&mut ptr, None, None) }
            .map_err(|e| error(e, "IMFMediaBuffer::Lock"))?;
        // SAFETY: the buffer is locked and was created `frame_bytes` long.
        let dst = unsafe { std::slice::from_raw_parts_mut(ptr, self.frame_bytes) };
        let copied = copy_source(format, size, src, origin, stride, dst, &self.planes);
        // SAFETY: locked above.
        let unlocked = unsafe { buffer.Unlock() };
        copied?;
        unlocked.map_err(|e| error(e, "IMFMediaBuffer::Unlock"))?;
        // SAFETY: a live buffer; the length is within its size.
        unsafe { buffer.SetCurrentLength(self.frame_bytes as u32) }
            .map_err(|e| error(e, "IMFMediaBuffer::SetCurrentLength"))?;
        Ok(sample)
    }

    /// A sample over a texture holding the region of `src` at `origin` for
    /// the frame at sample `time`, and whether it is the source's own
    /// texture, read in place.
    fn fill_texture(
        &mut self,
        src: &TensorDyn,
        origin: (usize, usize),
        time: i64,
    ) -> Result<(IMFSample, bool)> {
        let format = self.config.input_format;
        let size = (self.config.width as usize, self.config.height as usize);
        let in_flight = &self.in_flight;
        let textures = self
            .textures
            .as_mut()
            .expect("the texture path runs on a D3D11 encoder");
        let texture_source = is_native_nv12(src) && src.view_origin().is_none();
        let whole = origin == (0, 0) && src.width() == Some(size.0) && src.height() == Some(size.1);
        if texture_source && whole && textures.in_place == Some(true) {
            let texture = tensor_texture(src)?;
            // Submits whatever the source's producer queued, which the
            // encoder reads behind on the same device.
            // SAFETY: a live context.
            unsafe { textures.context.Flush() };
            return Ok((self.platform.texture_sample(&texture)?, true));
        }
        // A ring texture whose frame has come back, or a new one.
        let free = textures
            .ring
            .iter()
            .position(|(_, held)| !held.is_some_and(|t| in_flight.iter().any(|&(f, _)| f == t)));
        let slot = match free {
            Some(slot) => slot,
            None => {
                let texture = TensorDyn::image(
                    size.0,
                    size.1,
                    PixelFormat::Nv12,
                    DType::U8,
                    Some(TensorMemory::DmaBuf),
                    CpuAccess::Write,
                )?;
                textures.ring.push((texture, None));
                log::debug!("{}: {} source textures", self.device, textures.ring.len());
                textures.ring.len() - 1
            }
        };
        let scratch = &textures.ring[slot].0;
        let target = tensor_texture(scratch)?;
        if texture_source {
            let (x, y) = (origin.0 as u32, origin.1 as u32);
            let region = D3D11_BOX {
                left: x,
                top: y,
                front: 0,
                right: x + size.0 as u32,
                bottom: y + size.1 as u32,
                back: 1,
            };
            let source = tensor_texture(src)?;
            // SAFETY: both are live NV12 textures on HAL's device, the box is
            // inside the source on even coordinates (`check_source`), and the
            // ring texture is exactly its size.
            unsafe {
                textures.context.CopySubresourceRegion(
                    &target,
                    0,
                    0,
                    0,
                    0,
                    &source,
                    0,
                    Some(&region),
                )
            };
        } else {
            let stride = src
                .effective_row_stride()
                .ok_or_else(|| CodecError::InvalidInput("source has no row stride".into()))?;
            let pitch = scratch.effective_row_stride().ok_or_else(|| {
                CodecError::InvalidInput("a source texture has no row stride".into())
            })?;
            let to: Vec<Plane> = format
                .plane_table(size.0, size.1, pitch)
                .ok_or(CodecError::UnsupportedFormat(format))?
                .iter()
                .map(|p| Plane {
                    offset: p.offset as usize,
                    pitch: p.stride as usize,
                })
                .collect();
            let mut map = scratch.map_bytes(CpuAccess::Write)?;
            copy_source(format, size, src, origin, stride, &mut map, &to)?;
        }
        textures.ring[slot].1 = Some(time);
        // SAFETY: a live context.
        unsafe { textures.context.Flush() };
        Ok((self.platform.texture_sample(&target)?, false))
    }

    /// Waits, up to [`IN_PLACE_WAIT`], for the access unit of the frame at
    /// sample `time`. Returns whether it came.
    fn wait_for_frame(&mut self, time: i64) -> Result<bool> {
        let deadline = Instant::now() + IN_PLACE_WAIT;
        while self.in_flight.iter().any(|&(t, _)| t == time) {
            if Instant::now() >= deadline {
                return Ok(false);
            }
            let mut out = Vec::new();
            self.transform.collect(
                &self.platform,
                Duration::from_millis(2),
                &mut out,
                |out, _| !out.is_empty(),
            )?;
            self.take(out)?;
        }
        Ok(true)
    }

    /// Asks for the next frame to be an IDR picture.
    fn force_keyframe(&self, required: bool) -> Result<()> {
        let set = self.codec_api.as_ref().map(|api| {
            // SAFETY: a live codec interface and a VT_UI4 value.
            unsafe { api.SetValue(&CODECAPI_AVEncVideoForceKeyFrame, &VARIANT::from(1u32)) }
        });
        match set {
            Some(Ok(())) => Ok(()),
            _ if !required => {
                log::debug!("{}: cannot force a key frame", self.device);
                Ok(())
            }
            Some(Err(e)) => Err(CodecError::InvalidConfig(format!(
                "{} cannot force key frames: {}",
                self.device,
                e.message()
            ))),
            None => Err(CodecError::InvalidConfig(format!(
                "{} cannot force key frames",
                self.device
            ))),
        }
    }

    /// Turns the transform's outputs into access units.
    fn take(&mut self, outputs: Vec<Output>) -> Result<()> {
        for output in outputs {
            let Output::Sample(sample) = output else {
                continue;
            };
            // SAFETY: a live sample.
            let time = unsafe { sample.GetSampleTime() }.ok();
            // SAFETY: as above.
            let clean =
                unsafe { sample.GetUINT32(&MFSampleExtension_CleanPoint) }.is_ok_and(|v| v != 0);
            let bytes = sample_bytes(&sample)?;
            if !bytes.is_empty() {
                self.accept(bytes, clean, time);
            }
        }
        Ok(())
    }

    /// Queues an access unit, or holds a header-only sample for the next
    /// one.
    fn accept(&mut self, bytes: Vec<u8>, clean: bool, time: Option<i64>) {
        let au = AccessUnit::parse(&bytes);
        if au.ends_previous() {
            match self.pending.back_mut() {
                Some(previous) => previous.data.extend_from_slice(&bytes),
                None => log::debug!(
                    "{}: dropped {} bytes ending an access unit already returned",
                    self.device,
                    bytes.len()
                ),
            }
            return;
        }
        if !au.has_picture {
            self.headers.extend_from_slice(&bytes);
            return;
        }
        let pts = self.take_pts(time);
        let mut data = std::mem::take(&mut self.headers);
        let au = if data.is_empty() {
            data = bytes;
            au
        } else {
            data.extend_from_slice(&bytes);
            AccessUnit::parse(&data)
        };
        if au.parameter_sets {
            self.parameter_sets = h264::parameter_sets(&data);
        } else if au.idr && !self.parameter_sets.is_empty() {
            // Some encoders send the headers only on the first IDR, or in
            // the output type; every IDR gets them when headers repeat, and
            // the first one always does.
            let first = self.queued == 0;
            if self.config.repeat_headers || first {
                data = h264::with_parameter_sets(&data, &self.parameter_sets);
            }
        }
        self.queued += 1;
        self.pending.push_back(EncodedFrame {
            keyframe: clean || au.idr,
            data,
            pts,
        });
    }

    /// The PTS of the source whose sample time is `time`, dropping older
    /// sources the encoder skipped. Without a time, the oldest source's.
    fn take_pts(&mut self, time: Option<i64>) -> u64 {
        let Some(time) = time else {
            return self.in_flight.pop_front().map_or(0, |(_, pts)| pts);
        };
        while let Some(&(t, pts)) = self.in_flight.front() {
            self.in_flight.pop_front();
            if t == time {
                return pts;
            }
            if t > time {
                self.in_flight.push_front((t, pts));
                break;
            }
            log::debug!("{}: no access unit for the source at {t}", self.device);
        }
        log::warn!("{}: access unit with unknown time {time}", self.device);
        0
    }

    /// Forgets everything in flight after a failure; the next frame starts
    /// a new sequence.
    fn reset(&mut self) {
        self.transform.discard();
        self.in_flight.clear();
        self.headers.clear();
        self.restart = true;
        self.queued = 0;
    }
}

impl EncoderBackend for MfEncoder {
    fn encode(&mut self, src: &TensorDyn, opts: &FrameOptions) -> Result<Option<EncodedFrame>> {
        com();
        let origin = self.check_source(src, opts)?;
        let time = self.next_seq * self.duration;
        let (sample, in_place) = if self.textures.is_some() {
            self.fill_texture(src, origin, time)?
        } else {
            (self.fill(src, origin)?, false)
        };
        if opts.keyframe || self.restart {
            self.force_keyframe(opts.keyframe)?;
        }
        // SAFETY: a live sample.
        unsafe { sample.SetSampleTime(time) }.map_err(|e| error(e, "IMFSample::SetSampleTime"))?;
        // SAFETY: as above.
        unsafe { sample.SetSampleDuration(self.duration) }
            .map_err(|e| error(e, "IMFSample::SetSampleDuration"))?;
        self.next_seq += 1;
        self.in_flight.push_back((time, opts.pts));

        let mut out = Vec::new();
        let fed = self
            .transform
            .input(&self.platform, &sample, INPUT_TIMEOUT, &mut out);
        let taken = self.take(std::mem::take(&mut out));
        if let Err(e) = fed.and(taken) {
            self.reset();
            return Err(e);
        }
        self.restart = false;
        let measure = self.textures.as_ref().is_some_and(|t| t.in_place.is_none());
        if in_place || measure {
            // An encoder reads a texture after `ProcessInput` returns, so the
            // frame's own access unit is what says it has.
            let returned = match self.wait_for_frame(time) {
                Ok(returned) => returned,
                Err(e) => {
                    self.reset();
                    return Err(e);
                }
            };
            let textures = self.textures.as_mut().expect("a D3D11 encoder");
            if measure {
                textures.in_place = Some(returned);
                log::info!(
                    "{}: {}",
                    self.device,
                    if returned {
                        "returns access units at once; texture sources are read in place"
                    } else {
                        "holds frames; texture sources are copied on the GPU"
                    }
                );
            } else if !returned {
                textures.in_place = Some(false);
                log::warn!(
                    "{}: no access unit for a frame read in place within {IN_PLACE_WAIT:?}; \
                     draining it, and copying sources from now on",
                    self.device
                );
                // The caller may reuse the source once `encode` returns, so
                // the encoder must be done reading it: a drained encoder has
                // consumed every input, and a flushed one releases them.
                let drained = self
                    .transform
                    .drain(&self.platform, DRAIN_TIMEOUT, &mut out);
                if let Err(e) = drained.and_then(|()| self.take(std::mem::take(&mut out))) {
                    self.reset();
                    return Err(e);
                }
            }
        } else if self.pending.is_empty() {
            let collected = self.transform.collect(
                &self.platform,
                OUTPUT_GRACE,
                &mut out,
                |out, wants_input| !out.is_empty() || wants_input,
            );
            if let Err(e) = collected.and_then(|()| self.take(out)) {
                self.reset();
                return Err(e);
            }
        }
        Ok(self.pending.pop_front())
    }

    fn flush(&mut self) -> Result<Vec<EncodedFrame>> {
        com();
        let mut out = Vec::new();
        let drained = self
            .transform
            .drain(&self.platform, DRAIN_TIMEOUT, &mut out);
        let taken = self.take(out);
        if let Err(e) = drained.and(taken) {
            self.reset();
            return Err(e);
        }
        if !self.in_flight.is_empty() {
            log::debug!(
                "{}: {} sources had no access unit",
                self.device,
                self.in_flight.len()
            );
            self.in_flight.clear();
        }
        self.headers.clear();
        // The next frame opens a new sequence, as after a drain on V4L2.
        self.restart = true;
        self.queued = 0;
        Ok(self.pending.drain(..).collect())
    }

    fn device(&self) -> String {
        self.device.clone()
    }
}

/// Copies the `size` region of `src` at `origin`, whose rows are `stride`
/// bytes apart, into the planes `to` of `dst`.
fn copy_source(
    format: PixelFormat,
    size: (usize, usize),
    src: &TensorDyn,
    origin: (usize, usize),
    stride: usize,
    dst: &mut [u8],
    to: &[Plane],
) -> Result<()> {
    let read_error = |e: edgefirst_tensor::Error| {
        CodecError::InvalidInput(format!("the source cannot be read: {e}"))
    };
    let shapes =
        copy::plane_shapes(format, size.0, size.1).ok_or(CodecError::UnsupportedFormat(format))?;
    let luma = src.map_bytes(CpuAccess::Read).map_err(read_error)?;
    let copied = if src.is_multiplane() {
        let chroma = src.chroma_dyn().ok_or_else(|| {
            CodecError::InvalidInput("multi-plane source has no chroma plane".into())
        })?;
        let cstride = chroma.effective_row_stride().unwrap_or(stride);
        let cmap = chroma.map_bytes(CpuAccess::Read).map_err(read_error)?;
        let planes = copy::offset(
            format,
            &[
                Plane {
                    offset: 0,
                    pitch: stride,
                },
                Plane {
                    offset: 0,
                    pitch: cstride,
                },
            ],
            origin,
        );
        copy::copy_plane(&luma, planes[0], dst, to[0], shapes[0])
            .and_then(|()| copy::copy_plane(&cmap, planes[1], dst, to[1], shapes[1]))
    } else {
        let table = format
            .plane_table(src.width().unwrap_or(0), src.height().unwrap_or(0), stride)
            .ok_or(CodecError::UnsupportedFormat(format))?;
        let planes: Vec<Plane> = table
            .iter()
            .map(|p| Plane {
                offset: p.offset as usize,
                pitch: p.stride as usize,
            })
            .collect();
        copy::copy(
            format,
            size,
            &luma,
            &copy::offset(format, &planes, origin),
            dst,
            to,
        )
    };
    copied.map_err(CodecError::InvalidInput)
}

/// Sets the rate control, GOP, B-frame and latency properties `config`
/// asks for, where the encoder supports them.
fn apply_codec_api(api: &ICodecAPI, config: &EncoderConfig, name: &str) {
    let mut values: Vec<(&GUID, VARIANT, &str)> = vec![
        (
            &CODECAPI_AVEncCommonRateControlMode,
            VARIANT::from(eAVEncCommonRateControlMode_CBR.0 as u32),
            "constant bitrate",
        ),
        (
            &CODECAPI_AVEncCommonMeanBitRate,
            VARIANT::from(config.target_bitrate()),
            "bitrate",
        ),
        (
            &CODECAPI_AVEncMPVDefaultBPictureCount,
            VARIANT::from(0u32),
            "no B-frames",
        ),
        (
            &CODECAPI_AVLowLatencyMode,
            VARIANT::from(1u32),
            "low latency",
        ),
    ];
    if let Some(gop) = config.gop {
        values.push((&CODECAPI_AVEncMPVGOPSize, VARIANT::from(gop), "GOP"));
    }
    for (key, value, what) in values {
        // SAFETY: a live codec interface and a value of the property's type.
        if let Err(e) = unsafe { api.SetValue(key, &value) } {
            log::debug!("{name}: {what} not set: {}", e.message());
        }
    }
}

/// A video media type of `subtype` at the configured size and rate.
fn video_type(platform: &Platform, subtype: GUID, config: &EncoderConfig) -> Result<IMFMediaType> {
    let t = platform.media_type()?;
    let attrs: IMFAttributes = t.cast().map_err(|e| error(e, "IMFAttributes"))?;
    let set = |key: &GUID, value: u32| {
        // SAFETY: a live media type.
        unsafe { attrs.SetUINT32(key, value) }.map_err(|e| error(e, "IMFAttributes::SetUINT32"))
    };
    // SAFETY: a live media type.
    unsafe { attrs.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video) }
        .map_err(|e| error(e, "IMFAttributes::SetGUID"))?;
    // SAFETY: as above.
    unsafe { attrs.SetGUID(&MF_MT_SUBTYPE, &subtype) }
        .map_err(|e| error(e, "IMFAttributes::SetGUID"))?;
    set_pair(&attrs, &MF_MT_FRAME_SIZE, (config.width, config.height))?;
    set_pair(
        &attrs,
        &MF_MT_FRAME_RATE,
        frame_rate_ratio(config.frame_rate),
    )?;
    set_pair(&attrs, &MF_MT_PIXEL_ASPECT_RATIO, (1, 1))?;
    set(&MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive.0 as u32)?;
    Ok(t)
}

/// The H.264 output type, with `profile` and `level` when given.
fn output_type(
    platform: &Platform,
    config: &EncoderConfig,
    profile: Option<H264Profile>,
    level: Option<H264Level>,
) -> Result<IMFMediaType> {
    let t = video_type(platform, MFVideoFormat_H264, config)?;
    let attrs: IMFAttributes = t.cast().map_err(|e| error(e, "IMFAttributes"))?;
    let set = |key: &GUID, value: u32| {
        // SAFETY: a live media type.
        unsafe { attrs.SetUINT32(key, value) }.map_err(|e| error(e, "IMFAttributes::SetUINT32"))
    };
    set(&MF_MT_AVG_BITRATE, config.target_bitrate())?;
    if let Some(p) = profile {
        set(&MF_MT_MPEG2_PROFILE, profile_value(p))?;
    }
    if let Some(l) = level {
        set(&MF_MT_MPEG2_LEVEL, level_value(l))?;
    }
    Ok(t)
}

/// The bytes of every buffer of `sample`.
fn sample_bytes(sample: &IMFSample) -> Result<Vec<u8>> {
    // SAFETY: a live sample.
    let buffer = unsafe { sample.ConvertToContiguousBuffer() }
        .map_err(|e| error(e, "IMFSample::ConvertToContiguousBuffer"))?;
    let (mut ptr, mut len) = (std::ptr::null_mut(), 0u32);
    // SAFETY: valid out-pointers; unlocked below.
    unsafe { buffer.Lock(&mut ptr, None, Some(&mut len)) }
        .map_err(|e| error(e, "IMFMediaBuffer::Lock"))?;
    // SAFETY: the buffer is locked and holds `len` bytes.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len as usize) }.to_vec();
    // SAFETY: locked above.
    let _ = unsafe { buffer.Unlock() };
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_nv12_and_yuyv_are_taken() {
        assert_eq!(input_subtype(PixelFormat::Nv12), Some(MFVideoFormat_NV12));
        assert_eq!(input_subtype(PixelFormat::Yuyv), Some(MFVideoFormat_YUY2));
        assert_eq!(input_subtype(PixelFormat::Rgba), None);
        assert_eq!(input_subtype(PixelFormat::Nv16), None);
    }

    #[test]
    fn profiles_and_levels_use_the_codec_api_values() {
        assert_eq!(profile_value(H264Profile::Baseline), 66);
        assert_eq!(profile_value(H264Profile::ConstrainedBaseline), 256);
        assert_eq!(profile_value(H264Profile::Main), 77);
        assert_eq!(profile_value(H264Profile::High), 100);
        assert_eq!(level_value(H264Level::L1b), 11);
        assert_eq!(level_value(H264Level::L3_1), 31);
        assert_eq!(level_value(H264Level::L4_0), 40);
        assert_eq!(level_value(H264Level::L6_2), 62);
    }

    #[test]
    fn unset_profile_and_level_fall_back_to_the_encoder_default() {
        let config = EncoderConfig::h264(1920, 1080, PixelFormat::Nv12, 30.0);
        assert_eq!(
            output_attempts(&config),
            vec![
                (Some(H264Profile::High), Some(H264Level::L4_0)),
                (Some(H264Profile::High), None),
                (None, None),
            ]
        );
    }

    #[test]
    fn a_requested_profile_or_level_is_never_dropped() {
        let mut config = EncoderConfig::h264(1920, 1080, PixelFormat::Nv12, 30.0);
        config.profile = Some(H264Profile::Main);
        assert_eq!(
            output_attempts(&config),
            vec![
                (Some(H264Profile::Main), Some(H264Level::L4_0)),
                (Some(H264Profile::Main), None),
            ]
        );
        config.level = Some(H264Level::L4_1);
        assert_eq!(
            output_attempts(&config),
            vec![(Some(H264Profile::Main), Some(H264Level::L4_1))]
        );
        config.profile = None;
        assert_eq!(
            output_attempts(&config),
            vec![
                (Some(H264Profile::High), Some(H264Level::L4_1)),
                (None, Some(H264Level::L4_1)),
            ]
        );
    }

    #[test]
    fn packed_rows_have_no_padding() {
        assert_eq!(tight_pitch(PixelFormat::Nv12, 640), 640);
        assert_eq!(tight_pitch(PixelFormat::Yuyv, 640), 1280);
    }
}
