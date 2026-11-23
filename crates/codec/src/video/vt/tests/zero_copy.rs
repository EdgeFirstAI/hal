// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Encoding CoreVideo surfaces in place, how long the encoder holds them,
//! and how many copied frames it queues.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use super::*;
use crate::video::vt::color::from_attachments;
use crate::video::vt::encoder::{MAX_FRAME_DELAY, MAX_IN_FLIGHT, MAX_QUEUED_COPIES};

/// One frame period at 30 fps.
const PERIOD: Duration = Duration::from_nanos(1_000_000_000 / 30);

/// A CoreVideo pool of IOSurface-backed pixel buffers.
struct Pool(CfOwned);

impl Pool {
    fn new(w: usize, h: usize, fourcc: u32) -> Self {
        let number = |v: usize| cf_number_i32(v as i32).unwrap();
        let (format, width, height) = (number(fourcc as usize), number(w), number(h));
        // SAFETY: the dictionaries hold live CF objects, `pool` is a valid
        // out-pointer, and `CVPixelBufferPoolCreate` returns +1.
        unsafe {
            let surface = cf_dictionary(&[]).unwrap();
            let attributes = cf_dictionary(&[
                (kCVPixelBufferPixelFormatTypeKey, format.as_ptr()),
                (kCVPixelBufferWidthKey, width.as_ptr()),
                (kCVPixelBufferHeightKey, height.as_ptr()),
                (kCVPixelBufferIOSurfacePropertiesKey, surface.as_ptr()),
            ])
            .unwrap();
            let mut pool = std::ptr::null_mut();
            let status = CVPixelBufferPoolCreate(
                std::ptr::null(),
                std::ptr::null(),
                attributes.as_ptr(),
                &mut pool,
            );
            assert_eq!(status, kCVReturnSuccess);
            Self(CfOwned::from_create(pool).unwrap())
        }
    }

    /// The next pixel buffer the pool hands out.
    fn buffer(&self) -> CfOwned {
        let mut buffer = std::ptr::null_mut();
        // SAFETY: a live pool and a valid out-pointer; the buffer is +1.
        unsafe {
            let status =
                CVPixelBufferPoolCreatePixelBuffer(std::ptr::null(), self.0.as_ptr(), &mut buffer);
            assert_eq!(status, kCVReturnSuccess);
            CfOwned::from_create(buffer).unwrap()
        }
    }

    /// The surface of the next pixel buffer, wrapped as a tensor. The pool's
    /// pixel buffer is released; the tensor retains only the surface.
    fn take(&self) -> TensorDyn {
        wrap(&self.buffer())
    }

    /// The surfaces of `n` pixel buffers the pool hands out at once, so no
    /// two are the same, wrapped as tensors. The pool's pixel buffers are
    /// released.
    fn distinct(&self, n: usize) -> Vec<TensorDyn> {
        let buffers: Vec<CfOwned> = (0..n).map(|_| self.buffer()).collect();
        buffers.iter().map(wrap).collect()
    }
}

/// The surface of `buffer` wrapped as a tensor.
fn wrap(buffer: &CfOwned) -> TensorDyn {
    // SAFETY: the pixel buffer is live, and the tensor retains the surface.
    unsafe { TensorDyn::from_planar_iosurface(CVPixelBufferGetIOSurface(buffer.as_ptr()), None) }
        .unwrap()
}

/// Sets SMPTE C primaries, the BT.709 transfer function and the BT.601
/// matrix on `buffer`, to propagate to its surface.
fn attach_smpte_c(buffer: &CfOwned) {
    // SAFETY: a live pixel buffer, CoreVideo key and value constants.
    unsafe {
        for (key, value) in [
            (
                kCVImageBufferColorPrimariesKey,
                kCVImageBufferColorPrimaries_SMPTE_C,
            ),
            (
                kCVImageBufferTransferFunctionKey,
                kCVImageBufferTransferFunction_ITU_R_709_2,
            ),
            (
                kCVImageBufferYCbCrMatrixKey,
                kCVImageBufferYCbCrMatrix_ITU_R_601_4,
            ),
        ] {
            CVBufferSetAttachment(
                buffer.as_ptr(),
                key,
                value,
                kCVAttachmentMode_ShouldPropagate,
            );
        }
    }
}

/// The colorimetry the attachments of a new pixel buffer over the surface
/// of the `420v` tensor `src` describe.
fn surface_attachments(src: &TensorDyn) -> Colorimetry {
    let mut wrapper = std::ptr::null_mut();
    // SAFETY: a live surface and a valid out-pointer; the wrapper is +1 and
    // live while its attachments are read.
    unsafe {
        let status = CVPixelBufferCreateWithIOSurface(
            std::ptr::null(),
            surface(src),
            std::ptr::null(),
            &mut wrapper,
        );
        assert_eq!(status, kCVReturnSuccess);
        let wrapper = CfOwned::from_create(wrapper).unwrap();
        from_attachments(wrapper.as_ptr(), FOURCC_420V)
    }
}

/// A `420v` surface from `pool` with SMPTE C colour attachments.
fn smpte_c_surface(pool: &Pool) -> TensorDyn {
    let buffer = pool.buffer();
    attach_smpte_c(&buffer);
    wrap(&buffer)
}

fn surface(t: &TensorDyn) -> IOSurfaceRef {
    t.iosurface_ref().expect("an IOSurface tensor")
}

fn id(t: &TensorDyn) -> u32 {
    // SAFETY: the tensor keeps its surface live.
    unsafe { IOSurfaceGetID(surface(t)) }
}

fn in_use(t: &TensorDyn) -> bool {
    // SAFETY: the tensor keeps its surface live.
    unsafe { IOSurfaceIsInUse(surface(t)) != 0 }
}

fn use_count(t: &TensorDyn) -> i32 {
    // SAFETY: the tensor keeps its surface live.
    unsafe { IOSurfaceGetUseCount(surface(t)) }
}

/// The luma level of frame `i`.
fn level(i: usize) -> u8 {
    (16 + 7 * (i % 30)) as u8
}

/// Fills an NV12 or YUYV image with luma `luma` and neutral chroma.
fn paint(t: &TensorDyn, luma: u8) {
    let (w, h) = (t.width().unwrap(), t.height().unwrap());
    let stride = t.effective_row_stride().unwrap();
    if t.format() == Some(PixelFormat::Yuyv) {
        let mut px = t.map_bytes(CpuAccess::Write).unwrap();
        for y in 0..h {
            for pair in px[y * stride..y * stride + 2 * w].chunks_exact_mut(4) {
                pair.copy_from_slice(&[luma, 128, luma, 128]);
            }
        }
        return;
    }
    let mut px = t.map_bytes(CpuAccess::Write).unwrap();
    for y in 0..h {
        px[y * stride..y * stride + w].fill(luma);
    }
    match t.chroma_dyn() {
        Some(chroma) => {
            drop(px);
            let cstride = chroma.effective_row_stride().unwrap();
            let mut px = chroma.map_bytes(CpuAccess::Write).unwrap();
            for y in 0..h / 2 {
                px[y * cstride..y * cstride + w].fill(128);
            }
        }
        None => {
            for y in h..h + h / 2 {
                px[y * stride..y * stride + w].fill(128);
            }
        }
    }
}

fn at(pts: u64) -> FrameOptions {
    FrameOptions {
        pts,
        ..FrameOptions::default()
    }
}

fn config(w: usize, h: usize, format: PixelFormat) -> EncoderConfig {
    EncoderConfig::h264(w as u32, h as u32, format, 30.0)
}

/// Opens an encoder for `test` whose session reads sources in place, or
/// reports a skip naming the encoder when it does not.
fn open_zero_copy(test: &str, config: EncoderConfig) -> Option<encoder::VtEncoder> {
    let enc = open_encoder(test, config)?;
    if enc.bounded_delay_for_test() {
        return Some(enc);
    }
    assert_unbounded_is_software(&enc);
    report_skip(&format!(
        "{test} - {} does not bound its frame delay, so it reads no source in place",
        enc.info().device
    ));
    None
}

/// Fails on an Apple silicon Mac when `enc` is a hardware encoder whose
/// session does not bound its frame delay.
fn assert_unbounded_is_software(enc: &encoder::VtEncoder) {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        assert!(
            !enc.info().hardware,
            "the hardware session on encoder {} does not bound its frame delay",
            enc.encoder_id_for_test().as_deref().unwrap_or("(no ID)")
        );
    }
}

/// The route a CoreVideo source of the configured size takes on `enc`.
fn corevideo_route(enc: &encoder::VtEncoder) -> Transfer {
    if enc.bounded_delay_for_test() {
        Transfer::ZeroCopy
    } else {
        Transfer::CpuCopy
    }
}

/// The `p`th percentile of `samples`.
fn percentile(samples: &[Duration], p: f64) -> Duration {
    let mut sorted = samples.to_vec();
    sorted.sort();
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

/// Waits until `start + period * i`.
fn pace(start: Instant, i: usize) {
    let due = start + PERIOD * i as u32;
    if let Some(left) = due.checked_duration_since(Instant::now()) {
        std::thread::sleep(left);
    }
}

/// Checks that each decoded frame's mean luma is within 3 of `level(pts)`.
fn check_levels(test: &str, units: &[EncodedFrame], level: impl Fn(u64) -> u8) {
    let Some((_, frames)) = decode_all(test, units) else {
        return;
    };
    assert_eq!(frames.len(), units.len());
    for f in &frames {
        let want = f64::from(level(f.pts()));
        let got = mean_luma(f.tensor());
        assert!(
            (got - want).abs() <= 3.0,
            "frame {}: mean luma {got}, painted {want}",
            f.pts()
        );
    }
}

/// The mean absolute difference of the luma of two decoded frames.
fn luma_difference(a: &TensorDyn, b: &TensorDyn) -> f64 {
    let (w, h) = (a.width().unwrap(), a.height().unwrap());
    let (sa, sb) = (
        a.effective_row_stride().unwrap(),
        b.effective_row_stride().unwrap(),
    );
    let (pa, pb) = (
        a.map_bytes(CpuAccess::Read).unwrap(),
        b.map_bytes(CpuAccess::Read).unwrap(),
    );
    let sum: u64 = (0..h)
        .flat_map(|y| (0..w).map(move |x| (y, x)))
        .map(|(y, x)| u64::from(pa[y * sa + x].abs_diff(pb[y * sb + x])))
        .sum();
    sum as f64 / (w * h) as f64
}

/// What a zero-copy pool run measured.
#[derive(Default)]
struct Paced {
    units: Vec<EncodedFrame>,
    calls: Vec<Duration>,
    latency: Vec<Duration>,
    /// Refused `encode()` calls; when unpaced, each refused frame is
    /// submitted again.
    refused: usize,
    /// Frames whose source was released before `encode()` returned.
    done_in_encode: usize,
    most_held: usize,
    elapsed: Duration,
}

/// Encodes `frames` zero-copy frames from a `w`x`h` `420v` pool, releasing
/// each source after `encode()`. When `paced` is set, frames come at 30 fps
/// and a refused frame is skipped; otherwise they come as fast as the
/// encoder takes them, and a refused frame is submitted again after a short
/// wait. `check` sees every buffer the pool hands out. Each source the
/// encoder still holds when `encode()` returns must be in use.
fn pool_run(
    enc: &mut encoder::VtEncoder,
    (w, h): (usize, usize),
    frames: usize,
    paced: bool,
    mut check: impl FnMut(&TensorDyn),
) -> Paced {
    let pool = Pool::new(w, h, FOURCC_420V);
    let mut run = Paced::default();
    let mut submitted: BTreeMap<u64, Instant> = BTreeMap::new();
    let start = Instant::now();
    for i in 0..frames {
        if paced {
            pace(start, i);
        }
        let src = pool.take();
        check(&src);
        paint(&src, level(i));
        let pts = i as u64;
        let mut called = Instant::now();
        let result = encode_retry::retry(
            || {
                called = Instant::now();
                let result = enc.encode(&src, &at(pts));
                run.calls.push(called.elapsed());
                if matches!(result, Err(CodecError::Timeout(_))) {
                    run.refused += 1;
                }
                result
            },
            |e| !paced && matches!(e, CodecError::Timeout(_)),
        );
        match result {
            Ok(au) => {
                if !in_use(&src) {
                    assert!(
                        !enc.holds_for_test(pts),
                        "frame {i}: the encoder holds surface {} and it is not in use",
                        id(&src)
                    );
                    run.done_in_encode += 1;
                }
                submitted.insert(pts, called);
                run.most_held = run.most_held.max(enc.held_for_test());
                run.units.extend(au);
            }
            Err(CodecError::Timeout(_)) => {}
            Err(e) => panic!("encode {i}: {e}"),
        }
        for au in &run.units[run.latency.len()..] {
            run.latency.push(submitted[&au.pts].elapsed());
        }
    }
    run.units.extend(enc.flush().expect("flush"));
    run.elapsed = start.elapsed();
    assert_eq!(enc.info().transfer, Transfer::ZeroCopy);
    for au in &run.units[run.latency.len()..] {
        run.latency.push(submitted[&au.pts].elapsed());
    }
    run
}

#[test]
fn zero_copy_and_copy_sources_decode_to_the_same_pictures() {
    let test = "zero_copy_and_copy_sources_decode_to_the_same_pictures";
    let (w, h) = (1280, 720);
    let Some(mut enc) = open_zero_copy(test, config(w, h, PixelFormat::Nv12)) else {
        return;
    };
    let pool = Pool::new(w, h, FOURCC_420V);
    let mut zero_copy = Vec::new();
    for i in 0..30 {
        let src = pool.take();
        paint(&src, level(i));
        zero_copy.extend(encode_retrying(&mut enc, &src, &at(i as u64)));
    }
    zero_copy.extend(enc.flush().expect("flush"));
    assert_eq!(enc.info().transfer, Transfer::ZeroCopy);
    let src = image(w, h, PixelFormat::Nv12);
    let mut copied = Vec::new();
    for i in 0..30 {
        paint(&src, level(i));
        copied.extend(encode_retrying(&mut enc, &src, &at(i as u64)));
    }
    copied.extend(enc.flush().expect("flush"));
    assert_eq!(enc.info().transfer, Transfer::CpuCopy);
    assert_eq!((zero_copy.len(), copied.len()), (30, 30));
    let Some((_, a)) = decode_all(test, &zero_copy) else {
        return;
    };
    let Some((_, b)) = decode_all(test, &copied) else {
        return;
    };
    assert_eq!((a.len(), b.len()), (30, 30));
    for (a, b) in a.iter().zip(&b) {
        assert_eq!(a.pts(), b.pts());
        let d = luma_difference(a.tensor(), b.tensor());
        assert!(d <= 2.0, "frame {}: mean luma difference {d}", a.pts());
    }
}

#[test]
fn a_pool_never_hands_out_a_surface_the_encoder_is_reading() {
    let test = "a_pool_never_hands_out_a_surface_the_encoder_is_reading";
    let Some(mut enc) = open_zero_copy(test, config(1920, 1080, PixelFormat::Nv12)) else {
        return;
    };
    let mut handed_out = 0;
    let run = pool_run(&mut enc, (1920, 1080), 60, true, |src| {
        handed_out += 1;
        assert!(
            !in_use(src),
            "the pool handed out surface {} while it is in use",
            id(src)
        );
    });
    eprintln!(
        "{test}: {handed_out} buffers handed out, {} refused, {} released before encode() \
         returned, at most {} held",
        run.refused, run.done_in_encode, run.most_held
    );
    assert_eq!(run.units.len() + run.refused, 60);
    check_levels(test, &run.units, |pts| level(pts as usize));
}

#[test]
fn a_zero_copy_source_is_in_use_until_its_access_unit_is_out() {
    let test = "a_zero_copy_source_is_in_use_until_its_access_unit_is_out";
    let Some(mut enc) = open_zero_copy(test, config(1920, 1080, PixelFormat::Nv12)) else {
        return;
    };
    let watched = Pool::new(1920, 1080, FOURCC_420V).distinct(2);
    let (first, second) = (&watched[0], &watched[1]);
    let others = Pool::new(1920, 1080, FOURCC_420V);
    let opening = others.take();
    paint(&opening, level(0));
    let mut out: Vec<EncodedFrame> = enc
        .encode(&opening, &at(0))
        .expect("encode")
        .into_iter()
        .collect();
    paint(first, level(1));
    let before = use_count(first);
    assert!(!in_use(first), "use count {before} before encode()");
    out.extend(enc.encode(first, &at(1)).expect("encode"));
    let held = use_count(first);
    assert!(in_use(first), "use count {held} after encode()");
    let start = Instant::now();
    let mut i = 2;
    while !out.iter().any(|au| au.pts == 1) {
        assert!(
            i < 30,
            "no access unit for the first frame after {i} frames"
        );
        pace(start, i);
        let src = others.take();
        paint(&src, level(i));
        out.extend(enc.encode(&src, &at(i as u64)).expect("encode"));
        i += 1;
    }
    let after = use_count(first);
    eprintln!("{test}: use count {before} before encode(), {held} after, {after} once its access unit is out");
    assert!(
        !in_use(first),
        "use count {after} once its access unit is out"
    );
    paint(second, level(i));
    enc.encode(second, &at(i as u64)).expect("encode");
    let held = use_count(second);
    enc.flush().expect("flush");
    let flushed = use_count(second);
    eprintln!("{test}: second source use count {held} after encode(), {flushed} after flush()");
    assert!(!in_use(second), "use count {flushed} after flush()");
}

/// Times every `encode()` call of 90 zero-copy frames at 1920x1080, then at
/// 3840x2160, refusals included, and checks the 95th percentile is under
/// 5 ms. Paced runs count refused frames as drops; unpaced runs submit
/// every frame.
fn check_call_times(test: &str, paced: bool) {
    for (w, h) in [(1920, 1080), (3840, 2160)] {
        let Some(mut enc) = open_zero_copy(test, config(w, h, PixelFormat::Nv12)) else {
            return;
        };
        let run = pool_run(&mut enc, (w, h), 90, paced, |_| {});
        let (p50, p95) = (percentile(&run.calls, 0.5), percentile(&run.calls, 0.95));
        let refused = if paced { "drops" } else { "refusals retried" };
        eprintln!(
            "{test}: {w}x{h} encode() p50 {p50:?} p95 {p95:?}, {} {refused}, at most {} held, hardware {}",
            run.refused,
            run.most_held,
            enc.info().hardware
        );
        let dropped = if paced { run.refused } else { 0 };
        assert_eq!(run.units.len() + dropped, 90);
        assert!(p95 < Duration::from_millis(5), "{w}x{h}: p95 {p95:?}");
    }
}

#[test]
fn encode_returns_within_a_frame_interval_at_1080p_and_4k() {
    check_call_times(
        "encode_returns_within_a_frame_interval_at_1080p_and_4k",
        true,
    );
}

#[test]
fn encode_returns_at_once_when_frames_come_faster_than_they_encode() {
    check_call_times(
        "encode_returns_at_once_when_frames_come_faster_than_they_encode",
        false,
    );
}

#[test]
fn a_full_in_flight_window_refuses_the_next_frame_at_once() {
    let test = "a_full_in_flight_window_refuses_the_next_frame_at_once";
    let (w, h) = (3840, 2160);
    let Some(mut enc) = open_zero_copy(test, config(w, h, PixelFormat::Nv12)) else {
        return;
    };
    let pool = Pool::new(w, h, FOURCC_420V);
    let sources = pool.distinct(MAX_IN_FLIGHT + 4);
    for (i, src) in sources.iter().enumerate() {
        paint(src, level(i));
    }
    let mut refusal = None;
    for (i, src) in sources.iter().enumerate() {
        let called = Instant::now();
        match enc.encode(src, &at(i as u64)) {
            Ok(_) => {}
            Err(e) => {
                refusal = Some((i, e, called.elapsed()));
                break;
            }
        }
    }
    let (index, e, took) = refusal.expect("a refusal within the pool");
    eprintln!("{test}: refused at submission {index} of a {MAX_IN_FLIGHT} window in {took:?}: {e}");
    assert!(index >= MAX_IN_FLIGHT, "refused at submission {index}");
    assert!(matches!(e, CodecError::Timeout(_)), "{e}");
    assert!(took < Duration::from_millis(5), "the refusal took {took:?}");
    enc.flush().expect("flush");
}

#[test]
fn sustains_4k_at_30_fps_zero_copy() {
    let test = "sustains_4k_at_30_fps_zero_copy";
    let Some(mut enc) = open_zero_copy(test, config(3840, 2160, PixelFormat::Nv12)) else {
        return;
    };
    if !enc.info().hardware {
        report_skip(&format!("{test} - the encoder is not hardware"));
        return;
    }
    let run = pool_run(&mut enc, (3840, 2160), 150, true, |_| {});
    let (p50, p95) = (
        percentile(&run.latency, 0.5),
        percentile(&run.latency, 0.95),
    );
    eprintln!(
        "{test}: latency p50 {p50:?} p95 {p95:?}, {:?} to the last access unit, at most {} held",
        run.elapsed, run.most_held
    );
    assert_eq!(run.refused, 0);
    assert_eq!(run.units.len(), 150);
    assert!(
        run.elapsed <= Duration::from_millis(5500),
        "{:?}",
        run.elapsed
    );
}

#[test]
fn yuyv_from_a_yuvs_surface_is_read_in_place() {
    let test = "yuyv_from_a_yuvs_surface_is_read_in_place";
    let Some(mut enc) = open_zero_copy(test, config(640, 480, PixelFormat::Yuyv)) else {
        return;
    };
    let pool = Pool::new(640, 480, FOURCC_YUVS);
    let mut units = Vec::new();
    for i in 0..10 {
        let src = pool.take();
        paint(&src, level(i));
        units.extend(encode_retrying(&mut enc, &src, &at(i as u64)));
    }
    assert_eq!(enc.info().transfer, Transfer::ZeroCopy);
    units.extend(enc.flush().expect("flush"));
    assert_eq!(units.len(), 10);
    check_levels(test, &units, |pts| level(pts as usize));
}

/// The VUI colour and transfer of a 3-frame stream from `pool` buffers
/// tagged `colour`.
fn pool_colour(
    test: &str,
    pool: &Pool,
    w: usize,
    h: usize,
    colour: Option<Colorimetry>,
) -> Option<VuiColour> {
    let mut enc = open_encoder(test, config(w, h, PixelFormat::Nv12))?;
    let mut units = Vec::new();
    for i in 0..3 {
        let mut src = pool.take();
        src.set_colorimetry(colour);
        paint(&src, level(i));
        units.extend(encode_retrying(&mut enc, &src, &at(i as u64)));
    }
    assert_eq!(enc.info().transfer, corevideo_route(&enc));
    units.extend(enc.flush().expect("flush"));
    assert_eq!(units.len(), 3);
    Some(vui_colour(&units[0].data))
}

#[test]
fn a_video_range_surface_is_described_as_video_range() {
    let test = "a_video_range_surface_is_described_as_video_range";
    let bt709 = Colorimetry {
        space: Some(ColorSpace::Bt709),
        transfer: Some(ColorTransfer::Bt709),
        encoding: Some(ColorEncoding::Bt709),
        range: Some(ColorRange::Limited),
    };
    let pool = Pool::new(1920, 1080, FOURCC_420V);
    let Some(got) = pool_colour(test, &pool, 1920, 1080, Some(bt709)) else {
        return;
    };
    assert_eq!(got, (Some((1, 1, 1)), Some(false)));
}

#[test]
fn a_full_range_surface_is_described_as_full_range() {
    let test = "a_full_range_surface_is_described_as_full_range";
    let pool = Pool::new(640, 480, FOURCC_420F);
    let Some(got) = pool_colour(test, &pool, 640, 480, None) else {
        return;
    };
    assert_eq!(got, (Some((6, 1, 6)), Some(true)));
}

#[test]
fn a_tagged_range_that_disagrees_with_the_surface_is_refused() {
    let test = "a_tagged_range_that_disagrees_with_the_surface_is_refused";
    let Some(mut enc) = open_encoder(test, config(640, 480, PixelFormat::Nv12)) else {
        return;
    };
    let src = Pool::new(640, 480, FOURCC_420V)
        .take()
        .with_colorimetry(Colorimetry {
            range: Some(ColorRange::Full),
            ..Colorimetry::default()
        });
    let e = enc.encode(&src, &at(0)).unwrap_err();
    assert!(matches!(e, CodecError::InvalidInput(_)), "{e}");
    assert!(!in_use(&src));
}

#[test]
fn surface_colour_attachments_describe_an_untagged_source() {
    let test = "surface_colour_attachments_describe_an_untagged_source";
    let pool = Pool::new(1920, 1080, FOURCC_420V);
    let tagged = pool.buffer();
    attach_smpte_c(&tagged);
    let plain = pool.buffer();
    for (name, buffer, want) in [("with", tagged, (6, 1, 6)), ("without", plain, (1, 1, 1))] {
        let src = wrap(&buffer);
        drop(buffer);
        let seen = surface_attachments(&src);
        eprintln!("{test}: a surface {name} colour attachments, wrapped again: {seen:?}");
        let Some(mut enc) = open_zero_copy(test, config(1920, 1080, PixelFormat::Nv12)) else {
            return;
        };
        paint(&src, level(0));
        let mut units: Vec<EncodedFrame> = enc
            .encode(&src, &at(0))
            .expect("encode")
            .into_iter()
            .collect();
        units.extend(enc.flush().expect("flush"));
        let got = vui_colour(&units[0].data);
        eprintln!("{test}: a surface {name} colour attachments is described as {got:?}");
        let exposed = seen.space.is_some();
        assert_eq!(
            got.0,
            Some(if exposed { want } else { (1, 1, 1) }),
            "{name}"
        );
    }
}

/// The VUI colour of a 1-frame stream from `src` on `enc`, and the route
/// the frame took.
fn one_frame_colour(enc: &mut encoder::VtEncoder, src: &TensorDyn) -> (VuiColour, Transfer) {
    paint(src, level(0));
    let mut units: Vec<EncodedFrame> = encode_retrying(enc, src, &at(0)).into_iter().collect();
    let route = enc.info().transfer;
    units.extend(enc.flush().expect("flush"));
    (vui_colour(&units[0].data), route)
}

#[test]
fn surface_colour_attachments_describe_the_stream_on_either_route() {
    let test = "surface_colour_attachments_describe_the_stream_on_either_route";
    let (w, h) = (1920, 1080);
    let src = smpte_c_surface(&Pool::new(w, h, FOURCC_420V));
    let exposed = surface_attachments(&src).space.is_some();
    let Some(mut enc) = open_encoder(test, config(w, h, PixelFormat::Nv12)) else {
        return;
    };
    let (described, route) = one_frame_colour(&mut enc, &src);
    drop(enc);
    encoder::VtEncoder::refuse_frame_delay_for_test(true);
    let copied = open_encoder(test, config(w, h, PixelFormat::Nv12))
        .map(|mut enc| one_frame_colour(&mut enc, &src));
    encoder::VtEncoder::refuse_frame_delay_for_test(false);
    let Some((copied, copy_route)) = copied else {
        return;
    };
    eprintln!(
        "{test}: attachments exposed {exposed}; {route:?} gives {described:?}, {copy_route:?} gives {copied:?}"
    );
    assert_eq!(copy_route, Transfer::CpuCopy);
    assert_eq!(copied, described);
    if exposed {
        assert_eq!(copied, (Some((6, 1, 6)), Some(false)));
    }
}

#[test]
fn a_cropped_frame_from_a_tagged_surface_keeps_the_stream_colour() {
    let test = "a_cropped_frame_from_a_tagged_surface_keeps_the_stream_colour";
    let (w, h) = (1920, 1080);
    let Some(mut enc) = open_encoder(test, config(w, h, PixelFormat::Nv12)) else {
        return;
    };
    let pool = Pool::new(w, h, FOURCC_420V);
    let mut units = Vec::new();
    let mut exposed = true;
    for i in 0..6 {
        let src = smpte_c_surface(&pool);
        exposed &= surface_attachments(&src).space.is_some();
        paint(&src, level(i));
        let cropped = i % 2 == 1;
        let opts = FrameOptions {
            crop: cropped.then(|| Region::new(0, 0, w, h)),
            ..at(i as u64)
        };
        units.extend(encode_retrying(&mut enc, &src, &opts));
        let route = if cropped {
            Transfer::CpuCopy
        } else {
            corevideo_route(&enc)
        };
        assert_eq!(enc.info().transfer, route, "frame {i}");
    }
    units.extend(enc.flush().expect("flush"));
    assert_eq!(units.len(), 6);
    let got = vui_colour(&units[0].data);
    eprintln!("{test}: attachments exposed {exposed}; described as {got:?}");
    if exposed {
        assert_eq!(got, (Some((6, 1, 6)), Some(false)));
    }
    check_levels(test, &units, |pts| level(pts as usize));
}

#[test]
fn yuyv_from_a_yuvf_surface_is_read_in_place_as_full_range() {
    let test = "yuyv_from_a_yuvf_surface_is_read_in_place_as_full_range";
    let Some(mut enc) = open_encoder(test, config(640, 480, PixelFormat::Yuyv)) else {
        return;
    };
    let pool = Pool::new(640, 480, FOURCC_YUVF);
    let mut units = Vec::new();
    for i in 0..10 {
        let src = pool.take();
        paint(&src, level(i));
        units.extend(encode_retrying(&mut enc, &src, &at(i as u64)));
        assert_eq!(enc.info().transfer, corevideo_route(&enc), "frame {i}");
    }
    units.extend(enc.flush().expect("flush"));
    assert_eq!(units.len(), 10);
    assert_eq!(vui_colour(&units[0].data), (Some((6, 1, 6)), Some(true)));
    let Some((infos, frames)) = decode_all(test, &units) else {
        return;
    };
    assert_eq!(frames.len(), 10);
    assert_eq!(
        infos[0].colorimetry.and_then(|c| c.range),
        Some(ColorRange::Full)
    );
}

#[test]
fn a_yuvs_surface_tagged_full_range_is_refused() {
    let test = "a_yuvs_surface_tagged_full_range_is_refused";
    let Some(mut enc) = open_encoder(test, config(640, 480, PixelFormat::Yuyv)) else {
        return;
    };
    let src = Pool::new(640, 480, FOURCC_YUVS)
        .take()
        .with_colorimetry(Colorimetry {
            range: Some(ColorRange::Full),
            ..Colorimetry::default()
        });
    let e = enc.encode(&src, &at(0)).unwrap_err();
    assert!(matches!(e, CodecError::InvalidInput(_)), "{e}");
    assert!(!in_use(&src));
}

#[test]
fn a_yuvf_surface_tagged_limited_range_is_refused() {
    let test = "a_yuvf_surface_tagged_limited_range_is_refused";
    let Some(mut enc) = open_encoder(test, config(640, 480, PixelFormat::Yuyv)) else {
        return;
    };
    let src = Pool::new(640, 480, FOURCC_YUVF)
        .take()
        .with_colorimetry(Colorimetry {
            range: Some(ColorRange::Limited),
            ..Colorimetry::default()
        });
    let e = enc.encode(&src, &at(0)).unwrap_err();
    assert!(matches!(e, CodecError::InvalidInput(_)), "{e}");
    assert!(!in_use(&src));
}

/// A `w`x`h` IOSurface-backed `fourcc` pixel buffer whose rows are aligned
/// to `align` bytes.
fn aligned_buffer(w: usize, h: usize, fourcc: u32, align: usize) -> CfOwned {
    let align = cf_number_i32(align as i32).unwrap();
    // SAFETY: the dictionaries hold live CF objects, `buffer` is a valid
    // out-pointer, and `CVPixelBufferCreate` returns +1.
    unsafe {
        let surface = cf_dictionary(&[]).unwrap();
        let attributes = cf_dictionary(&[
            (kCVPixelBufferIOSurfacePropertiesKey, surface.as_ptr()),
            (kCVPixelBufferBytesPerRowAlignmentKey, align.as_ptr()),
        ])
        .unwrap();
        let mut buffer = std::ptr::null_mut();
        let status = CVPixelBufferCreate(
            std::ptr::null(),
            w,
            h,
            fourcc,
            attributes.as_ptr(),
            &mut buffer,
        );
        assert_eq!(status, kCVReturnSuccess);
        CfOwned::from_create(buffer).unwrap()
    }
}

/// The surface of the YUYV pixel buffer `buffer` wrapped as a YUYV tensor
/// with tightly packed rows.
fn wrap_tight_yuyv(buffer: &CfOwned) -> TensorDyn {
    // SAFETY: the pixel buffer is live, and the tensor retains the surface.
    unsafe {
        let surface = CVPixelBufferGetIOSurface(buffer.as_ptr());
        let (w, h) = (IOSurfaceGetWidth(surface), IOSurfaceGetHeight(surface));
        let mut t = TensorDyn::from_iosurface(surface, &[h, w, 2], DType::U8, None).unwrap();
        t.set_format(PixelFormat::Yuyv).unwrap();
        t
    }
}

#[test]
fn a_source_whose_row_stride_is_not_the_surface_pitch_is_copied() {
    let test = "a_source_whose_row_stride_is_not_the_surface_pitch_is_copied";
    let (w, h) = (600, 480);
    let buffer = aligned_buffer(w, h, FOURCC_YUVS, 256);
    // SAFETY: a live pixel buffer.
    let pitch = unsafe { CVPixelBufferGetBytesPerRow(buffer.as_ptr()) };
    let tight = wrap_tight_yuyv(&buffer);
    let pitched = wrap(&buffer);
    drop(buffer);
    let stride = tight.effective_row_stride().unwrap();
    eprintln!("{test}: surface pitch {pitch}, tensor row stride {stride}");
    assert_ne!(stride, pitch);
    let Some(mut enc) = open_encoder(test, config(w, h, PixelFormat::Yuyv)) else {
        return;
    };
    let mut units = Vec::new();
    for i in 0..10 {
        paint(&tight, level(i));
        units.extend(encode_retrying(&mut enc, &tight, &at(i as u64)));
        assert_eq!(enc.info().transfer, Transfer::CpuCopy, "frame {i}");
        assert!(!in_use(&tight), "frame {i}");
    }
    units.extend(enc.flush().expect("flush"));
    assert_eq!(units.len(), 10);
    check_levels(test, &units, |pts| level(pts as usize));
    let Some(mut enc) = open_encoder(test, config(w, h, PixelFormat::Yuyv)) else {
        return;
    };
    assert_eq!(pitched.effective_row_stride(), Some(pitch));
    let route = corevideo_route(&enc);
    paint(&pitched, level(0));
    let mut units: Vec<EncodedFrame> = encode_retrying(&mut enc, &pitched, &at(0))
        .into_iter()
        .collect();
    assert_eq!(enc.info().transfer, route);
    units.extend(enc.flush().expect("flush"));
    check_levels(test, &units, |pts| level(pts as usize));
}

#[test]
fn a_cropped_planar_source_is_copied() {
    let test = "a_cropped_planar_source_is_copied";
    let Some(mut enc) = open_encoder(test, config(1280, 720, PixelFormat::Nv12)) else {
        return;
    };
    let pool = Pool::new(1920, 1080, FOURCC_420V);
    let mut units = Vec::new();
    for i in 0..10 {
        let src = pool.take();
        paint(&src, level(i));
        let opts = FrameOptions {
            crop: Some(Region::new(320, 180, 1280, 720)),
            ..at(i as u64)
        };
        units.extend(encode_retrying(&mut enc, &src, &opts));
        assert!(!in_use(&src), "frame {i}");
    }
    assert_eq!(enc.info().transfer, Transfer::CpuCopy);
    units.extend(enc.flush().expect("flush"));
    assert_eq!(units.len(), 10);
}

/// Encodes 4 zero-copy 1280x720 frames, keeping their sources.
fn four_held(enc: &mut encoder::VtEncoder) -> Vec<TensorDyn> {
    let sources = Pool::new(1280, 720, FOURCC_420V).distinct(4);
    for (i, src) in sources.iter().enumerate() {
        paint(src, level(i));
        enc.encode(src, &at(i as u64)).expect("encode");
    }
    sources
}

#[test]
fn dropping_the_encoder_releases_every_source() {
    let test = "dropping_the_encoder_releases_every_source";
    let Some(mut enc) = open_zero_copy(test, config(1280, 720, PixelFormat::Nv12)) else {
        return;
    };
    let sources = four_held(&mut enc);
    assert_eq!(enc.info().transfer, Transfer::ZeroCopy);
    drop(enc);
    for (i, src) in sources.iter().enumerate() {
        assert!(!in_use(src), "source {i}: use count {}", use_count(src));
    }
}

#[test]
fn a_session_error_releases_every_source() {
    let test = "a_session_error_releases_every_source";
    let Some(mut enc) = open_zero_copy(test, config(1280, 720, PixelFormat::Nv12)) else {
        return;
    };
    let sources = four_held(&mut enc);
    assert_eq!(enc.info().transfer, Transfer::ZeroCopy);
    enc.invalidate_session_for_test();
    let pool = Pool::new(1280, 720, FOURCC_420V);
    let failed = pool.take();
    paint(&failed, level(4));
    let e = enc
        .encode(&failed, &at(4))
        .and_then(|_| enc.flush().map(drop))
        .unwrap_err();
    assert!(matches!(e, CodecError::Device { .. }), "{e}");
    for (i, src) in sources.iter().chain([&failed]).enumerate() {
        assert!(!in_use(src), "source {i}: use count {}", use_count(src));
    }
    let next = pool.take();
    paint(&next, level(5));
    let mut out: Vec<EncodedFrame> = enc
        .encode(&next, &at(5))
        .expect("encode")
        .into_iter()
        .collect();
    out.extend(enc.flush().expect("flush"));
    let after = out
        .iter()
        .find(|f| f.pts == 5)
        .expect("the frame after the error");
    assert_eq!(parameter_sets_and_idr(&after.data), (true, true));
    assert!(after.keyframe);
}

#[test]
fn a_full_copy_queue_refuses_the_next_frame_without_taking_a_pool_buffer() {
    let test = "a_full_copy_queue_refuses_the_next_frame_without_taking_a_pool_buffer";
    let Some(mut enc) = open_encoder(test, config(1920, 1080, PixelFormat::Nv12)) else {
        return;
    };
    let src = image(1920, 1080, PixelFormat::Nv12);
    let mut units = Vec::new();
    enc.pause_submission_for_test(true);
    for i in 0..MAX_QUEUED_COPIES {
        paint(&src, level(i));
        units.extend(enc.encode(&src, &at(i as u64)).expect("encode"));
    }
    let taken = enc.pool_buffers_for_test();
    let refused = MAX_QUEUED_COPIES;
    paint(&src, level(refused));
    let called = Instant::now();
    let e = enc.encode(&src, &at(refused as u64)).unwrap_err();
    let took = called.elapsed();
    eprintln!("{test}: refused with {MAX_QUEUED_COPIES} queued in {took:?}: {e}");
    assert!(matches!(e, CodecError::Timeout(_)), "{e}");
    assert!(took < Duration::from_millis(5), "the refusal took {took:?}");
    assert_eq!(
        enc.pool_buffers_for_test(),
        taken,
        "a pool buffer was taken"
    );
    enc.pause_submission_for_test(false);
    for i in refused..20 {
        paint(&src, level(i));
        units.extend(encode_retrying(&mut enc, &src, &at(i as u64)));
    }
    units.extend(enc.flush().expect("flush"));
    let pts: Vec<u64> = units.iter().map(|au| au.pts).collect();
    assert_eq!(pts, (0..20).collect::<Vec<u64>>());
    check_levels(test, &units, |pts| level(pts as usize));
}

#[test]
fn a_session_holds_at_most_max_frame_delay_frames() {
    let test = "a_session_holds_at_most_max_frame_delay_frames";
    let (w, h) = (1280, 720);
    let Some(mut enc) = open_zero_copy(test, config(w, h, PixelFormat::Nv12)) else {
        return;
    };
    let pool = Pool::new(w, h, FOURCC_420V);
    let mut first_out = None;
    let mut held = Vec::new();
    for i in 0..3 * MAX_IN_FLIGHT {
        let src = pool.take();
        paint(&src, level(i));
        encode_retrying(&mut enc, &src, &at(i as u64));
        let deadline = Instant::now() + Duration::from_secs(2);
        while enc.held_for_test() > MAX_FRAME_DELAY && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        std::thread::sleep(Duration::from_millis(50));
        let now = enc.held_for_test();
        if first_out.is_none() && now <= i {
            first_out = Some(i + 1);
        }
        held.push(now);
    }
    enc.flush().expect("flush");
    eprintln!(
        "{test}: hardware {}, {first_out:?} frames submitted before the first output, held {held:?}",
        enc.info().hardware
    );
    assert!(held.iter().all(|&n| n <= MAX_FRAME_DELAY), "held {held:?}");
}

#[test]
fn a_session_without_a_bounded_frame_delay_copies_corevideo_sources() {
    let test = "a_session_without_a_bounded_frame_delay_copies_corevideo_sources";
    let (w, h) = (1280, 720);
    encoder::VtEncoder::refuse_frame_delay_for_test(true);
    let Some(mut enc) = open_encoder(test, config(w, h, PixelFormat::Nv12)) else {
        return;
    };
    assert!(!enc.bounded_delay_for_test());
    let pool = Pool::new(w, h, FOURCC_420V);
    let mut units = Vec::new();
    for i in 0..10 {
        let src = pool.take();
        paint(&src, level(i));
        units.extend(encode_retrying(&mut enc, &src, &at(i as u64)));
        assert!(!in_use(&src), "frame {i}: use count {}", use_count(&src));
        assert_eq!(enc.held_for_test(), 0, "frame {i}");
        assert_eq!(enc.info().transfer, Transfer::CpuCopy, "frame {i}");
    }
    units.extend(enc.flush().expect("flush"));
    encoder::VtEncoder::refuse_frame_delay_for_test(false);
    assert_eq!(units.len(), 10);
    check_levels(test, &units, |pts| level(pts as usize));
}

#[test]
fn corevideo_sources_take_the_route_the_session_allows() {
    let test = "corevideo_sources_take_the_route_the_session_allows";
    let (w, h) = (1280, 720);
    let Some(mut enc) = open_encoder(test, config(w, h, PixelFormat::Nv12)) else {
        return;
    };
    if !enc.bounded_delay_for_test() {
        assert_unbounded_is_software(&enc);
    }
    let route = corevideo_route(&enc);
    eprintln!("{test}: {} takes {route:?}", enc.info().device);
    let pool = Pool::new(w, h, FOURCC_420V);
    let mut units = Vec::new();
    for i in 0..10 {
        let src = pool.take();
        paint(&src, level(i));
        units.extend(encode_retrying(&mut enc, &src, &at(i as u64)));
        assert_eq!(enc.info().transfer, route, "frame {i}");
        if route == Transfer::CpuCopy {
            assert!(!in_use(&src), "frame {i}: use count {}", use_count(&src));
            assert_eq!(enc.held_for_test(), 0, "frame {i}");
        }
    }
    units.extend(enc.flush().expect("flush"));
    assert_eq!(units.len(), 10);
    check_levels(test, &units, |pts| level(pts as usize));
}
