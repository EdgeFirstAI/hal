// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Tracker C API - Multi-object tracking with ByteTrack.
//!
//! This module provides ByteTrack multi-object tracking functionality.

use crate::decoder::HalDetectBoxList;
use crate::error::set_error;
use crate::{check_null, check_null_ret_null};
use edgefirst_decoder::DetectBox;
use edgefirst_tracker::bytetrack::ByteTrack;
use edgefirst_tracker::{DetectionBox, TrackInfo, Tracker};
use libc::{c_int, size_t};
use uuid::Uuid;

/// Track information for a tracked object.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HalTrackInfo {
    /// Unique identifier for this track (16 bytes UUID)
    pub uuid: [u8; 16],
    /// Predicted location in XYXY format (xmin, ymin, xmax, ymax)
    pub location: [f32; 4],
    /// Number of times this track has been updated
    pub count: i32,
    /// Timestamp when this track was first created (nanoseconds)
    pub created: u64,
    /// Timestamp of the last update (nanoseconds)
    pub last_updated: u64,
}

impl From<&TrackInfo> for HalTrackInfo {
    fn from(info: &TrackInfo) -> Self {
        Self {
            uuid: *info.uuid.as_bytes(),
            location: info.tracked_location,
            count: info.count,
            created: info.created,
            last_updated: info.last_updated,
        }
    }
}

/// Opaque ByteTrack tracker type.
pub struct HalByteTrack {
    inner: ByteTrack,
}

/// List of track info results.
pub struct HalTrackInfoList {
    tracks: Vec<TrackInfo>,
}

/// Internal wrapper to implement DetectionBox trait
#[derive(Debug)]
struct DetectBoxWrapper<'a> {
    inner: &'a DetectBox,
}

impl<'a> DetectionBox for DetectBoxWrapper<'a> {
    fn bbox(&self) -> [f32; 4] {
        [
            self.inner.bbox.xmin,
            self.inner.bbox.ymin,
            self.inner.bbox.xmax,
            self.inner.bbox.ymax,
        ]
    }

    fn score(&self) -> f32 {
        self.inner.score
    }

    fn label(&self) -> usize {
        self.inner.label
    }
}

// ============================================================================
// ByteTrack Functions
// ============================================================================

/// Create a new ByteTrack tracker with specified parameters.
///
/// @param track_thresh Score threshold for creating new tracks
/// @param high_thresh High confidence threshold for first-pass matching
/// @param match_thresh IOU threshold for matching detections to tracks
/// @param frame_rate Expected frame rate of the input video
/// @param track_buffer Number of frames to keep lost tracks before deletion
/// @return New ByteTrack handle on success, NULL on error
/// @par Errors (errno):
/// - ENOMEM: Memory allocation failed
#[no_mangle]
pub extern "C" fn hal_bytetrack_new(
    track_thresh: f32,
    high_thresh: f32,
    match_thresh: f32,
    frame_rate: c_int,
    track_buffer: c_int,
) -> *mut HalByteTrack {
    let mut tracker = ByteTrack::new();
    tracker.track_update = track_thresh;
    tracker.track_high_conf = high_thresh;
    tracker.track_iou = match_thresh;
    // Convert frame rate and buffer to lifespan in nanoseconds
    // track_extra_lifespan = track_buffer frames * (1/frame_rate seconds/frame) * 1e9 ns/s
    if frame_rate > 0 {
        tracker.track_extra_lifespan =
            ((track_buffer as f64 / frame_rate as f64) * 1_000_000_000.0) as u64;
    }

    Box::into_raw(Box::new(HalByteTrack { inner: tracker }))
}

/// Create a new ByteTrack tracker with default parameters.
///
/// Default values:
/// - track_thresh: 0.25
/// - high_thresh: 0.7
/// - match_thresh: 0.25
/// - track_extra_lifespan: 500ms
///
/// @return New ByteTrack handle on success, NULL on error
/// @par Errors (errno):
/// - ENOMEM: Memory allocation failed
#[no_mangle]
pub extern "C" fn hal_bytetrack_new_default() -> *mut HalByteTrack {
    Box::into_raw(Box::new(HalByteTrack {
        inner: ByteTrack::new(),
    }))
}

/// Update the tracker with new detections.
///
/// @param tracker ByteTrack handle
/// @param detections Detection box list from decoder
/// @param timestamp Current timestamp in nanoseconds
/// @return Track info list with updated tracks (caller must free), NULL on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL tracker/detections)
#[no_mangle]
pub unsafe extern "C" fn hal_bytetrack_update(
    tracker: *mut HalByteTrack,
    detections: *const HalDetectBoxList,
    timestamp: u64,
) -> *mut HalTrackInfoList {
    check_null_ret_null!(tracker, detections);

    let boxes = &(*detections).boxes;
    let wrappers: Vec<DetectBoxWrapper> = boxes.iter().map(|b| DetectBoxWrapper { inner: b }).collect();

    let results = (*tracker).inner.update(&wrappers, timestamp);

    // Collect non-None track infos
    let tracks: Vec<TrackInfo> = results.into_iter().flatten().collect();

    Box::into_raw(Box::new(HalTrackInfoList { tracks }))
}

/// Get all currently active tracks.
///
/// @param tracker ByteTrack handle
/// @return Track info list (caller must free), NULL on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL tracker)
#[no_mangle]
pub unsafe extern "C" fn hal_bytetrack_get_active_tracks(
    tracker: *const HalByteTrack,
) -> *mut HalTrackInfoList {
    check_null_ret_null!(tracker);

    let tracks = <ByteTrack as Tracker<DetectBoxWrapper>>::get_active_tracks(&(*tracker).inner);
    Box::into_raw(Box::new(HalTrackInfoList { tracks }))
}

/// Free a ByteTrack tracker.
///
/// @param tracker ByteTrack handle to free (can be NULL, no-op)
#[no_mangle]
pub unsafe extern "C" fn hal_bytetrack_free(tracker: *mut HalByteTrack) {
    if !tracker.is_null() {
        drop(Box::from_raw(tracker));
    }
}

// ============================================================================
// Track Info List Functions
// ============================================================================

/// Get the number of tracks in a list.
///
/// @param list Track info list handle
/// @return Number of tracks, or 0 if list is NULL
#[no_mangle]
pub unsafe extern "C" fn hal_track_info_list_len(list: *const HalTrackInfoList) -> size_t {
    if list.is_null() {
        return 0;
    }
    (*list).tracks.len()
}

/// Get a track info from a list by index.
///
/// @param list Track info list handle
/// @param index Index of the track (0-based)
/// @param out_info Output parameter for the track info
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL list/out_info, index out of bounds)
#[no_mangle]
pub unsafe extern "C" fn hal_track_info_list_get(
    list: *const HalTrackInfoList,
    index: size_t,
    out_info: *mut HalTrackInfo,
) -> c_int {
    check_null!(list, out_info);

    if index >= (*list).tracks.len() {
        return set_error(libc::EINVAL);
    }

    *out_info = HalTrackInfo::from(&(&(*list).tracks)[index]);
    0
}

/// Free a track info list.
///
/// @param list Track info list handle to free (can be NULL, no-op)
#[no_mangle]
pub unsafe extern "C" fn hal_track_info_list_free(list: *mut HalTrackInfoList) {
    if !list.is_null() {
        drop(Box::from_raw(list));
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Convert a UUID byte array to a string representation.
///
/// The output buffer must be at least 37 bytes (36 chars + null terminator).
///
/// @param uuid UUID byte array (16 bytes)
/// @param out_str Output buffer for string representation
/// @param out_len Size of output buffer
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL: Invalid argument (NULL pointers, buffer too small)
#[no_mangle]
pub unsafe extern "C" fn hal_uuid_to_string(
    uuid: *const [u8; 16],
    out_str: *mut libc::c_char,
    out_len: size_t,
) -> c_int {
    check_null!(uuid, out_str);

    if out_len < 37 {
        return set_error(libc::EINVAL);
    }

    let uuid_obj = Uuid::from_bytes(*uuid);
    let uuid_str = uuid_obj.to_string();

    std::ptr::copy_nonoverlapping(uuid_str.as_ptr() as *const libc::c_char, out_str, 36);
    *out_str.add(36) = 0; // null terminator

    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytetrack_create_and_free() {
        unsafe {
            let tracker = hal_bytetrack_new_default();
            assert!(!tracker.is_null());
            hal_bytetrack_free(tracker);
        }
    }

    #[test]
    fn test_bytetrack_with_params() {
        unsafe {
            let tracker = hal_bytetrack_new(0.3, 0.8, 0.2, 30, 60);
            assert!(!tracker.is_null());
            hal_bytetrack_free(tracker);
        }
    }

    #[test]
    fn test_null_handling() {
        unsafe {
            assert_eq!(hal_track_info_list_len(std::ptr::null()), 0);
            hal_track_info_list_free(std::ptr::null_mut());
            hal_bytetrack_free(std::ptr::null_mut());
        }
    }

    #[test]
    fn test_uuid_to_string() {
        unsafe {
            let uuid = [0u8; 16];
            let mut buffer = [0i8; 37];
            let result = hal_uuid_to_string(&uuid, buffer.as_mut_ptr(), 37);
            assert_eq!(result, 0);
            // UUID of all zeros should be "00000000-0000-0000-0000-000000000000"
            let s = std::ffi::CStr::from_ptr(buffer.as_ptr());
            assert_eq!(s.to_str().unwrap(), "00000000-0000-0000-0000-000000000000");
        }
    }
}
