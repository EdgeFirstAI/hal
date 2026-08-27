// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! ByteTrack: associating detections across frames.

use std::ffi::{c_char, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};

use edgefirst_decoder_abi::EfDetectBox;
use edgefirst_tracker::bytetrack::ByteTrack;
use edgefirst_tracker::{ByteTrackBuilder, DetectionBox, TrackInfo, Tracker};

/// Plain detection value for the C surface. Tracker does not depend on
/// `edgefirst-tensor`; it only needs six scalars per box.
#[derive(Debug, Clone, Copy)]
struct CDetectBox {
    bbox: [f32; 4],
    score: f32,
    label: usize,
}

impl DetectionBox for CDetectBox {
    fn bbox(&self) -> [f32; 4] {
        self.bbox
    }
    fn score(&self) -> f32 {
        self.score
    }
    fn label(&self) -> usize {
        self.label
    }
}

/// An opaque tracker.
pub struct EfByteTrack {
    inner: ByteTrack<CDetectBox>,
}

/// One track: a stable identity, its predicted location, and its history.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct EfTrackInfo {
    /// Stable identity for this track, as raw UUID bytes.
    ///
    /// Bytes rather than a string so the struct stays a plain value with no
    /// allocation; format it with [`ef_uuid_to_string`] when a human needs it.
    pub uuid: [u8; 16],
    /// Smoothed predicted location, XYXY normalized.
    pub location: [f32; 4],
    /// How many times this track has been updated.
    pub count: i32,
    /// Nanosecond timestamp when the track was created.
    pub created: u64,
    /// Nanosecond timestamp of the most recent update.
    pub last_updated: u64,
}

impl From<&TrackInfo> for EfTrackInfo {
    fn from(t: &TrackInfo) -> Self {
        Self {
            uuid: *t.uuid.as_bytes(),
            location: t.tracked_location,
            count: t.count,
            created: t.created,
            last_updated: t.last_updated,
        }
    }
}

/// An owned list of tracks.
pub struct EfTrackInfoList {
    tracks: Vec<TrackInfo>,
}

/// Create a tracker with explicit parameters.
///
/// @param track_update  Smoothness of track updates; higher is more stable,
///                      lower more responsive.
/// @param high_thresh   Confidence above which a detection can start a track.
/// @param match_thresh  IoU required to match a detection to a track.
#[no_mangle]
pub extern "C" fn ef_bytetrack_new(
    track_update: f32,
    high_thresh: f32,
    match_thresh: f32,
) -> *mut EfByteTrack {
    catch_unwind(|| {
        Box::into_raw(Box::new(EfByteTrack {
            inner: ByteTrackBuilder::new()
                .track_update(track_update)
                .track_high_conf(high_thresh)
                .track_iou(match_thresh)
                .build(),
        }))
    })
    .unwrap_or(std::ptr::null_mut())
}

/// Create a tracker with the library's defaults.
#[no_mangle]
pub extern "C" fn ef_bytetrack_new_default() -> *mut EfByteTrack {
    catch_unwind(|| {
        Box::into_raw(Box::new(EfByteTrack {
            inner: ByteTrackBuilder::new().build(),
        }))
    })
    .unwrap_or(std::ptr::null_mut())
}

/// Free a tracker. Freeing `NULL` is a no-op.
///
/// # Safety
/// `t` must be `NULL` or have come from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_bytetrack_free(t: *mut EfByteTrack) {
    unsafe {
        if t.is_null() {
            return;
        }
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(t))));
    }
}

/// Advance the tracker by one frame.
///
/// `timestamp` is in nanoseconds and must be monotonic; the tracker uses it for
/// track age and expiry, so a non-monotonic value corrupts both.
///
/// @return a track list the caller must free, or `NULL` on a null argument.
///
/// # Safety
/// `detections` must point to `count` readable elements, or be `NULL` when
/// `count` is 0 — obtain them from `ef_detect_box_list_data`.
#[no_mangle]
pub unsafe extern "C" fn ef_bytetrack_update(
    t: *mut EfByteTrack,
    detections: *const EfDetectBox,
    count: usize,
    timestamp: u64,
) -> *mut EfTrackInfoList {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if t.is_null() || (detections.is_null() && count != 0) {
                return std::ptr::null_mut();
            }
            // A borrowed C array, not a handle: passing an opaque list across would
            // force this library to link the one that owns it.
            let boxes: Vec<CDetectBox> = if count == 0 {
                Vec::new()
            } else {
                std::slice::from_raw_parts(detections, count)
                    .iter()
                    .map(|b| CDetectBox {
                        bbox: [b.xmin, b.ymin, b.xmax, b.ymax],
                        score: b.score,
                        label: b.label as usize,
                    })
                    .collect()
            };
            let results = (*t).inner.update(&boxes, timestamp);
            let tracks: Vec<TrackInfo> = results.into_iter().flatten().collect();
            Box::into_raw(Box::new(EfTrackInfoList { tracks }))
        }))
        .unwrap_or(std::ptr::null_mut())
    }
}

/// Every track currently alive, without advancing the tracker.
///
/// @return a track list the caller must free, or `NULL`.
///
/// # Safety
/// `t` must be live.
#[no_mangle]
pub unsafe extern "C" fn ef_bytetrack_active_tracks(t: *const EfByteTrack) -> *mut EfTrackInfoList {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if t.is_null() {
                return std::ptr::null_mut();
            }
            // `get_active_tracks` returns the richer ActiveTrackInfo; the C
            // surface exposes the smoothed TrackInfo, matching `update`.
            let tracks: Vec<TrackInfo> = (*t)
                .inner
                .get_active_tracks()
                .into_iter()
                .map(|a| a.info)
                .collect();
            Box::into_raw(Box::new(EfTrackInfoList { tracks }))
        }))
        .unwrap_or(std::ptr::null_mut())
    }
}

/// Number of tracks. Zero for a `NULL` list.
///
/// # Safety
/// `l` must be `NULL` or valid.
#[no_mangle]
pub unsafe extern "C" fn ef_track_info_list_len(l: *const EfTrackInfoList) -> usize {
    unsafe {
        if l.is_null() {
            return 0;
        }
        catch_unwind(AssertUnwindSafe(|| {
            let tracks = &(*l).tracks;
            tracks.len()
        }))
        .unwrap_or(0)
    }
}

/// Copy track `index` into `out`.
///
/// Copies rather than lending a pointer: a borrowed element dangles the moment
/// the list is freed, and C gives the caller no way to notice.
///
/// @return 0 on success, `EINVAL` for a null argument or out-of-range index.
///
/// # Safety
/// `out` must be writable.
#[no_mangle]
pub unsafe extern "C" fn ef_track_info_list_get(
    l: *const EfTrackInfoList,
    index: usize,
    out: *mut EfTrackInfo,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if l.is_null() || out.is_null() {
                return libc::EINVAL;
            }
            let tracks = &(*l).tracks;
            match tracks.get(index) {
                Some(t) => {
                    *out = EfTrackInfo::from(t);
                    0
                }
                None => libc::EINVAL,
            }
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Free a track list. Freeing `NULL` is a no-op.
///
/// # Safety
/// `l` must be `NULL` or have come from this library.
#[no_mangle]
pub unsafe extern "C" fn ef_track_info_list_free(l: *mut EfTrackInfoList) {
    unsafe {
        if l.is_null() {
            return;
        }
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(l))));
    }
}

/// Format a track's UUID into `out` as 36 characters plus a NUL.
///
/// `out` must have room for 37 bytes. Writing into a caller's buffer rather
/// than returning an allocation keeps ownership with the caller and needs no
/// matching free.
///
/// @return 0 on success, `EINVAL` on a null argument.
///
/// # Safety
/// `uuid` must be 16 readable bytes; `out` must be 37 writable bytes.
#[no_mangle]
pub unsafe extern "C" fn ef_uuid_to_string(uuid: *const u8, out: *mut c_char) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if uuid.is_null() || out.is_null() {
                return libc::EINVAL;
            }
            let mut bytes = [0u8; 16];
            bytes.copy_from_slice(std::slice::from_raw_parts(uuid, 16));
            let s = uuid::Uuid::from_bytes(bytes).hyphenated().to_string();
            std::ptr::copy_nonoverlapping(s.as_ptr() as *const c_char, out, s.len());
            *out.add(s.len()) = 0;
            0
        }))
        .unwrap_or(libc::EINVAL)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tracker_can_be_created_and_freed() {
        unsafe {
            let t = ef_bytetrack_new_default();
            assert!(!t.is_null());
            ef_bytetrack_free(t);
            let t = ef_bytetrack_new(0.5, 0.6, 0.7);
            assert!(!t.is_null());
            ef_bytetrack_free(t);
            ef_bytetrack_free(std::ptr::null_mut());
        }
    }

    #[test]
    fn a_detection_becomes_a_track_with_a_stable_identity() {
        // The property that matters: the same object across two frames keeps
        // one UUID. A per-frame list would satisfy a length check but not this.
        unsafe {
            let t = ef_bytetrack_new_default();
            let mut first = [0u8; 16];
            for frame in 0..2u64 {
                let dets = [EfDetectBox {
                    xmin: 0.10,
                    ymin: 0.10,
                    xmax: 0.30,
                    ymax: 0.30,
                    score: 0.95,
                    label: 1,
                }];
                let list = ef_bytetrack_update(t, dets.as_ptr(), dets.len(), frame * 33_000_000);
                assert!(!list.is_null());
                if frame == 1 {
                    assert_eq!(ef_track_info_list_len(list), 1, "one object, one track");
                    let mut info = EfTrackInfo::default();
                    assert_eq!(ef_track_info_list_get(list, 0, &mut info), 0);
                    assert_ne!(info.uuid, [0u8; 16], "a track must have an identity");
                    first = info.uuid;
                }
                ef_track_info_list_free(list);
            }
            let active = ef_bytetrack_active_tracks(t);
            if ef_track_info_list_len(active) > 0 {
                let mut info = EfTrackInfo::default();
                ef_track_info_list_get(active, 0, &mut info);
                assert_eq!(info.uuid, first, "identity must persist across frames");
            }
            ef_track_info_list_free(active);
            ef_bytetrack_free(t);
        }
    }

    #[test]
    fn null_arguments_are_errors_not_crashes() {
        unsafe {
            let mut out = EfTrackInfo::default();
            assert!(ef_bytetrack_update(std::ptr::null_mut(), std::ptr::null(), 0, 0).is_null());
            // A null array with a non-zero count is a caller bug, not an
            // empty frame: refuse rather than read.
            let t = ef_bytetrack_new_default();
            assert!(ef_bytetrack_update(t, std::ptr::null(), 5, 0).is_null());
            // A genuinely empty frame is legal — objects can leave the scene.
            let empty = ef_bytetrack_update(t, std::ptr::null(), 0, 0);
            assert!(!empty.is_null(), "an empty detection set is a valid frame");
            ef_track_info_list_free(empty);
            ef_bytetrack_free(t);
            assert!(ef_bytetrack_active_tracks(std::ptr::null()).is_null());
            assert_eq!(ef_track_info_list_len(std::ptr::null()), 0);
            assert_eq!(
                ef_track_info_list_get(std::ptr::null(), 0, &mut out),
                libc::EINVAL
            );
            ef_track_info_list_free(std::ptr::null_mut());
        }
    }

    #[test]
    fn a_uuid_formats_to_36_characters_plus_nul() {
        unsafe {
            let bytes = [0xABu8; 16];
            // c_char is i8 on macOS and u8 on aarch64-linux; naming the type
            // keeps this compiling on both.
            let mut buf = [0 as c_char; 37];
            assert_eq!(ef_uuid_to_string(bytes.as_ptr(), buf.as_mut_ptr()), 0);
            let s = std::ffi::CStr::from_ptr(buf.as_ptr()).to_str().unwrap();
            assert_eq!(s.len(), 36, "hyphenated UUID is 36 chars");
            assert_eq!(s.matches('-').count(), 4);
            assert_eq!(
                ef_uuid_to_string(std::ptr::null(), buf.as_mut_ptr()),
                libc::EINVAL
            );
        }
    }
}
