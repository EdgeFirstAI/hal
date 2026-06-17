// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Multi-object tracking for EdgeFirst inference pipelines.
//!
//! This crate provides the [`Tracker`] trait and a concrete implementation,
//! [`ByteTrack`], which associates per-frame detections
//! into persistent tracks using two-pass IoU matching and Kalman-smoothed
//! trajectory prediction.
//!
//! # Quick start
//!
//! ```rust
//! use edgefirst_tracker::{bytetrack::ByteTrackBuilder, Tracker, MockDetection};
//!
//! let mut tracker = ByteTrackBuilder::new().build();
//!
//! // Timestamps are nanoseconds; here we use small integers for clarity.
//! let frame1 = vec![MockDetection::new([0.1, 0.1, 0.5, 0.5], 0.9, 0)];
//! let infos = tracker.update(&frame1, 1_000_000);
//!
//! // Each element corresponds to the detection at the same index; `Some` means
//! // the detection was either matched to an existing track or started a new one.
//! if let Some(info) = infos[0] {
//!     println!("track {} created at t={}", info.uuid, info.created);
//! }
//! ```
//!
//! # Terminology
//!
//! - **Tracklet** — an internal state machine (Kalman filter + metadata) that
//!   follows one object across frames.
//! - **Detection** — a single-frame bounding box with confidence score and
//!   class label, produced by a model decoder.
//! - **Track** — the public view of a tracklet: a [`TrackInfo`] value returned
//!   by [`Tracker::update`] or [`Tracker::get_active_tracks`].
//!
//! # Architecture
//!
//! See [`crates/tracker/ARCHITECTURE.md`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tracker/ARCHITECTURE.md)
//! for the full design description, Mermaid class diagram, tracking flow, and
//! Kalman filter state documentation.

use std::fmt::Debug;

pub use uuid::Uuid;
pub mod bytetrack;
pub use bytetrack::{ByteTrack, ByteTrackBuilder};
pub mod kalman;

/// A bounding box produced by a detection model, suitable for use with a [`Tracker`].
///
/// Implement this trait on your model's output type to feed it directly into
/// [`ByteTrack`] without copying. The tracker crate has
/// no dependency on `edgefirst-decoder`; any type that can report an XYXY box,
/// a confidence score, and a class label satisfies this contract.
///
/// # Coordinates
///
/// `bbox()` must return `[xmin, ymin, xmax, ymax]` in normalised `[0, 1]`
/// coordinates or in pixel coordinates — the tracker does not prescribe units,
/// but all detections passed to a single [`Tracker::update`] call must use the
/// same coordinate space.
///
/// # Example
///
/// ```rust
/// use edgefirst_tracker::DetectionBox;
///
/// #[derive(Debug, Clone)]
/// struct MyDetection { bbox: [f32; 4], score: f32, label: usize }
///
/// impl DetectionBox for MyDetection {
///     fn bbox(&self)  -> [f32; 4] { self.bbox }
///     fn score(&self) -> f32      { self.score }
///     fn label(&self) -> usize    { self.label }
/// }
/// ```
pub trait DetectionBox: Debug + Clone {
    /// Bounding box in `[xmin, ymin, xmax, ymax]` (XYXY) format.
    fn bbox(&self) -> [f32; 4];

    /// Confidence score in `[0, 1]`. Used by ByteTrack to separate
    /// high-confidence detections (first-pass matching) from low-confidence
    /// ones (second-pass matching).
    fn score(&self) -> f32;

    /// Class index (0-based). Stored in the tracklet for callers that need
    /// per-class track metadata; the tracker itself does not gate matching
    /// on label — a tracklet may receive detections from different classes.
    fn label(&self) -> usize;
}

/// A minimal detection box for unit tests and doc examples.
///
/// This type is **not** intended for production use. It is `#[doc(hidden)]` in
/// the public docs but available as `edgefirst_tracker::MockDetection` so test
/// code in downstream crates does not need to define its own dummy type.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct MockDetection {
    bbox: [f32; 4],
    score: f32,
    label: usize,
}

impl MockDetection {
    /// Creates a new mock detection with the given bounding box, score, and label.
    pub fn new(bbox: [f32; 4], score: f32, label: usize) -> Self {
        Self { bbox, score, label }
    }
}

impl DetectionBox for MockDetection {
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

/// Per-track metadata returned by [`Tracker::update`] and [`Tracker::get_active_tracks`].
///
/// Every active track carries a stable [`Uuid`] assigned at track creation and
/// preserved until the track is deleted. Callers can use the UUID to correlate
/// tracks across frames or to drive per-track rendering (e.g. per-UUID colour
/// assignment in `ColorMode::Track`).
///
/// # Timestamps
///
/// `created` and `last_updated` store the raw `timestamp` argument passed to
/// [`Tracker::update`] at the corresponding event. The tracker does not
/// interpret these values — they are caller-supplied nanoseconds in typical
/// usage, but any monotonically increasing unit works.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackInfo {
    /// Globally unique identifier for this track, assigned when the tracklet is
    /// first created and never reused.
    pub uuid: Uuid,

    /// Kalman-smoothed predicted location of the object in XYXY format
    /// `[xmin, ymin, xmax, ymax]`. This is the **predicted** position for the
    /// current frame, not the raw detection box — it may differ from the
    /// detection that triggered the update. For the raw detection see
    /// [`ActiveTrackInfo::last_box`].
    pub tracked_location: [f32; 4],

    /// Number of times this track has been matched to a detection. Starts at 1
    /// on the frame the track is created. Useful as a confidence signal —
    /// tracks with a low count are newly created and may be false positives.
    pub count: i32,

    /// Timestamp (as supplied to [`Tracker::update`]) of the frame on which
    /// this track was first created.
    pub created: u64,

    /// Timestamp of the most recent [`Tracker::update`] call that matched a
    /// detection to this track. Tracklets whose `last_updated` is older than
    /// `now - extra_lifespan` are deleted on the next update.
    pub last_updated: u64,
}

/// A [`TrackInfo`] paired with the most recent raw detection box for that track.
///
/// Returned by [`Tracker::get_active_tracks`] for callers that need both the
/// Kalman-smoothed position (in `info.tracked_location`) and the unsmoothed,
/// model-output bounding box (`last_box`) — for example, to draw the exact
/// predicted anchor alongside the smoothed track centre.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActiveTrackInfo<T: DetectionBox> {
    /// Smoothed track metadata.
    pub info: TrackInfo,

    /// The last raw detection box associated with this track, as returned by
    /// [`DetectionBox::bbox`] — before any Kalman smoothing. This reflects the
    /// model's output on the most recent frame the track was matched, not the
    /// current Kalman prediction.
    pub last_box: T,
}

/// Multi-object tracker interface.
///
/// Implementations associate a stream of per-frame detection boxes into
/// persistent tracks. The tracker owns all internal state (tracklet filters,
/// timestamps, IDs) and is mutable — `update` takes `&mut self`.
///
/// # Contract
///
/// - The returned `Vec` from `update` is parallel to the input `boxes` slice:
///   `result[i]` is `Some(info)` when detection `i` was matched to a track or
///   spawned a new one, and `None` when it was too low-confidence to initiate or
///   continue a track.
/// - `get_active_tracks` returns every tracklet that has not yet expired,
///   including those not matched in the most recent frame (they remain alive
///   until `extra_lifespan` elapses without a match).
/// - Callers must not share a single `Tracker` across concurrent threads without
///   external synchronization — `update` takes `&mut self`.
pub trait Tracker<T: DetectionBox> {
    /// Process one frame of detections and advance all internal tracklet states.
    ///
    /// `boxes` is the set of detections for this frame. `timestamp` is a
    /// monotonically increasing value (typically nanoseconds from a wall clock
    /// or pipeline counter) used for tracklet expiry.
    ///
    /// Returns a `Vec<Option<TrackInfo>>` of the same length as `boxes`.
    fn update(&mut self, boxes: &[T], timestamp: u64) -> Vec<Option<TrackInfo>>;

    /// Return all currently active tracklets, including those not matched in
    /// the most recent frame but not yet expired.
    fn get_active_tracks(&self) -> Vec<ActiveTrackInfo<T>>;
}
