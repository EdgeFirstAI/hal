// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::{
    kalman::ConstantVelocityXYAHModel2, ActiveTrackInfo, DetectionBox, TrackInfo, Tracker,
};
use lapjv::{lapjv, Matrix};
use log::trace;
use nalgebra::{Dyn, OMatrix, U4};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ByteTrackBuilder {
    track_extra_lifespan: u64,
    track_high_conf: f32,
    track_iou: f32,
    track_update: f32,
}

impl Default for ByteTrackBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ByteTrackBuilder {
    /// Creates a new ByteTrackBuilder with default parameters.
    /// These defaults are:
    /// - track_high_conf: 0.7
    /// - track_iou: 0.25
    /// - track_update: 0.25
    /// - track_extra_lifespan: 500_000_000 (0.5 seconds)
    /// # Examples
    /// ```rust
    /// use edgefirst_tracker::{bytetrack::ByteTrackBuilder, Tracker, MockDetection};
    /// let mut tracker = ByteTrackBuilder::new().build();
    /// assert_eq!(tracker.track_high_conf, 0.7);
    /// assert_eq!(tracker.track_iou, 0.25);
    /// assert_eq!(tracker.track_update, 0.25);
    /// assert_eq!(tracker.track_extra_lifespan, 500_000_000);
    /// # let boxes = Vec::<MockDetection>::new();
    /// # tracker.update(&boxes, 0);
    /// ```
    pub fn new() -> Self {
        Self {
            track_high_conf: 0.7,
            track_iou: 0.25,
            track_update: 0.25,
            track_extra_lifespan: 500_000_000,
        }
    }

    /// Sets the extra lifespan for tracks in nanoseconds.
    pub fn track_extra_lifespan(mut self, lifespan: u64) -> Self {
        self.track_extra_lifespan = lifespan;
        self
    }

    /// Sets the high confidence threshold for tracking.
    pub fn track_high_conf(mut self, conf: f32) -> Self {
        self.track_high_conf = conf;
        self
    }

    /// Sets the IOU threshold for tracking.
    pub fn track_iou(mut self, iou: f32) -> Self {
        self.track_iou = iou;
        self
    }

    /// Sets the update rate for the Kalman filter.
    pub fn track_update(mut self, update: f32) -> Self {
        self.track_update = update;
        self
    }

    /// Builds the ByteTrack tracker with the specified parameters.
    /// # Examples
    /// ```rust
    /// use edgefirst_tracker::{bytetrack::ByteTrackBuilder, Tracker, MockDetection};
    /// let mut tracker = ByteTrackBuilder::new()
    ///     .track_high_conf(0.8)
    ///     .track_iou(0.3)
    ///     .track_update(0.2)
    ///     .track_extra_lifespan(1_000_000_000)
    ///     .build();
    /// assert_eq!(tracker.track_high_conf, 0.8);
    /// assert_eq!(tracker.track_iou, 0.3);
    /// assert_eq!(tracker.track_update, 0.2);
    /// assert_eq!(tracker.track_extra_lifespan, 1_000_000_000);
    /// # let boxes = Vec::<MockDetection>::new();
    /// # tracker.update(&boxes, 0);
    /// ```
    pub fn build<T: DetectionBox>(self) -> ByteTrack<T> {
        ByteTrack {
            track_extra_lifespan: self.track_extra_lifespan,
            track_high_conf: self.track_high_conf,
            track_iou: self.track_iou,
            track_update: self.track_update,
            tracklets: Vec::new(),
            frame_count: 0,
        }
    }
}

#[allow(dead_code)]
#[derive(Default, Debug, Clone)]
pub struct ByteTrack<T: DetectionBox> {
    pub track_extra_lifespan: u64,
    pub track_high_conf: f32,
    pub track_iou: f32,
    pub track_update: f32,
    pub tracklets: Vec<Tracklet<T>>,
    pub frame_count: i32,
}

#[derive(Debug, Clone)]
pub struct Tracklet<T: DetectionBox> {
    pub id: Uuid,
    pub filter: ConstantVelocityXYAHModel2<f32>,
    pub count: i32,
    pub created: u64,
    pub last_updated: u64,
    pub last_box: T,
}

impl<T: DetectionBox> Tracklet<T> {
    fn update(&mut self, detect_box: &T, ts: u64) {
        self.count += 1;
        self.last_updated = ts;
        self.filter.update(&xyxy_to_xyah(&detect_box.bbox()));
        self.last_box = detect_box.clone();
    }

    pub fn get_predicted_location(&self) -> [f32; 4] {
        let projected = self.filter.project().0;
        let predicted_xyah = projected.as_slice();
        xyah_to_xyxy(predicted_xyah)
    }
}

fn xyxy_to_xyah(vaal_box: &[f32; 4]) -> [f32; 4] {
    let x = (vaal_box[2] + vaal_box[0]) / 2.0;
    let y = (vaal_box[3] + vaal_box[1]) / 2.0;
    let w = (vaal_box[2] - vaal_box[0]).max(EPSILON);
    let h = (vaal_box[3] - vaal_box[1]).max(EPSILON);
    let a = w / h;

    [x, y, a, h]
}

fn xyah_to_xyxy(xyah: &[f32]) -> [f32; 4] {
    assert!(xyah.len() >= 4);
    let [x, y, a, h] = xyah[0..4] else {
        unreachable!()
    };
    let w = h * a;
    [x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0]
}

const INVALID_MATCH: f32 = 1000000.0;
const EPSILON: f32 = 0.00001;

fn iou(box1: &[f32], box2: &[f32]) -> f32 {
    let intersection = (box1[2].min(box2[2]) - box1[0].max(box2[0])).max(0.0)
        * (box1[3].min(box2[3]) - box1[1].max(box2[1])).max(0.0);

    let union = (box1[2] - box1[0]) * (box1[3] - box1[1])
        + (box2[2] - box2[0]) * (box2[3] - box2[1])
        - intersection;

    if union <= EPSILON {
        return 0.0;
    }

    intersection / union
}

fn box_cost<T: DetectionBox>(
    track: &Tracklet<T>,
    new_box: &T,
    distance: f32,
    score_threshold: f32,
    iou_threshold: f32,
) -> f32 {
    let _ = distance;

    if new_box.score() < score_threshold {
        return INVALID_MATCH;
    }

    // use iou between predicted box and real box:
    let predicted_xyah = track.filter.mean.as_slice();
    let expected = xyah_to_xyxy(predicted_xyah);
    let iou = iou(&expected, &new_box.bbox());
    if iou < iou_threshold {
        return INVALID_MATCH;
    }
    (1.5 - new_box.score()) + (1.5 - iou)
}

impl<T: DetectionBox> ByteTrack<T> {
    fn compute_costs(
        &mut self,
        boxes: &[T],
        score_threshold: f32,
        iou_threshold: f32,
        box_filter: &[bool],
        track_filter: &[bool],
    ) -> Matrix<f32> {
        // costs matrix must be square
        let dims = boxes.len().max(self.tracklets.len());
        let mut measurements = OMatrix::<f32, Dyn, U4>::from_element(boxes.len(), 0.0);
        for (i, mut row) in measurements.row_iter_mut().enumerate() {
            row.copy_from_slice(&xyxy_to_xyah(&boxes[i].bbox()));
        }

        // TODO: use matrix math for IOU, should speed up computation, and store it in
        // distances

        Matrix::from_shape_fn((dims, dims), |(x, y)| {
            if x < boxes.len() && y < self.tracklets.len() {
                if box_filter[x] || track_filter[y] {
                    INVALID_MATCH
                } else {
                    box_cost(
                        &self.tracklets[y],
                        &boxes[x],
                        // distances[(x, y)],
                        0.0,
                        score_threshold,
                        iou_threshold,
                    )
                }
            } else {
                0.0
            }
        })
    }

    /// Process assignments from linear assignment and update tracking state.
    /// Returns true if any matches were made.
    #[allow(clippy::too_many_arguments)]
    fn process_assignments(
        &mut self,
        assignments: &[usize],
        boxes: &[T],
        costs: &Matrix<f32>,
        matched: &mut [bool],
        tracked: &mut [bool],
        matched_info: &mut [Option<TrackInfo>],
        timestamp: u64,
        log_assignments: bool,
    ) {
        for (i, &x) in assignments.iter().enumerate() {
            if i >= boxes.len() || x >= self.tracklets.len() {
                continue;
            }

            // Filter out invalid assignments
            if costs[(i, x)] >= INVALID_MATCH {
                continue;
            }

            // Skip already matched boxes/tracklets
            if matched[i] || tracked[x] {
                continue;
            }

            if log_assignments {
                trace!(
                    "Cost: {} Box: {:#?} UUID: {} Mean: {}",
                    costs[(i, x)],
                    boxes[i],
                    self.tracklets[x].id,
                    self.tracklets[x].filter.mean
                );
            }

            matched[i] = true;
            matched_info[i] = Some(TrackInfo {
                uuid: self.tracklets[x].id,
                count: self.tracklets[x].count,
                created: self.tracklets[x].created,
                tracked_location: self.tracklets[x].get_predicted_location(),
                last_updated: timestamp,
            });
            tracked[x] = true;
            self.tracklets[x].update(&boxes[i], timestamp);
        }
    }

    /// Remove expired tracklets based on timestamp.
    fn remove_expired_tracklets(&mut self, timestamp: u64) {
        // must iterate from the back
        for i in (0..self.tracklets.len()).rev() {
            let expiry = self.tracklets[i].last_updated + self.track_extra_lifespan;
            if expiry < timestamp {
                trace!("Tracklet removed: {:?}", self.tracklets[i].id);
                let _ = self.tracklets.swap_remove(i);
            }
        }
    }

    /// Create new tracklets from unmatched high-confidence boxes.
    fn create_new_tracklets(
        &mut self,
        boxes: &[T],
        high_conf_indices: &[usize],
        matched: &[bool],
        matched_info: &mut [Option<TrackInfo>],
        timestamp: u64,
    ) {
        for &i in high_conf_indices {
            if matched[i] {
                continue;
            }

            let id = Uuid::new_v4();
            let new_tracklet = Tracklet {
                id,
                filter: ConstantVelocityXYAHModel2::new(
                    &xyxy_to_xyah(&boxes[i].bbox()),
                    self.track_update,
                ),
                last_updated: timestamp,
                count: 1,
                created: timestamp,
                last_box: boxes[i].clone(),
            };
            matched_info[i] = Some(TrackInfo {
                uuid: new_tracklet.id,
                count: new_tracklet.count,
                created: new_tracklet.created,
                tracked_location: new_tracklet.get_predicted_location(),
                last_updated: timestamp,
            });
            self.tracklets.push(new_tracklet);
        }
    }
}

impl<T> Tracker<T> for ByteTrack<T>
where
    T: DetectionBox,
{
    fn update(&mut self, boxes: &[T], timestamp: u64) -> Vec<Option<TrackInfo>> {
        self.frame_count += 1;

        // Identify high-confidence detections
        let high_conf_ind: Vec<usize> = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| b.score() >= self.track_high_conf)
            .map(|(x, _)| x)
            .collect();

        let mut matched = vec![false; boxes.len()];
        let mut tracked = vec![false; self.tracklets.len()];
        let mut matched_info = vec![None; boxes.len()];

        // First pass: match high-confidence detections
        if !self.tracklets.is_empty() {
            for track in &mut self.tracklets {
                track.filter.predict();
            }

            let costs = self.compute_costs(
                boxes,
                self.track_high_conf,
                self.track_iou,
                &matched,
                &tracked,
            );
            if let Ok(ans) = lapjv(&costs) {
                self.process_assignments(
                    &ans.0,
                    boxes,
                    &costs,
                    &mut matched,
                    &mut tracked,
                    &mut matched_info,
                    timestamp,
                    false,
                );
            }
        }

        // Second pass: match remaining tracklets to low-confidence detections
        if !self.tracklets.is_empty() {
            let costs = self.compute_costs(boxes, 0.0, self.track_iou, &matched, &tracked);
            if let Ok(ans) = lapjv(&costs) {
                self.process_assignments(
                    &ans.0,
                    boxes,
                    &costs,
                    &mut matched,
                    &mut tracked,
                    &mut matched_info,
                    timestamp,
                    true,
                );
            }
        }

        // Remove expired tracklets
        self.remove_expired_tracklets(timestamp);

        // Create new tracklets from unmatched high-confidence boxes
        self.create_new_tracklets(
            boxes,
            &high_conf_ind,
            &matched,
            &mut matched_info,
            timestamp,
        );

        matched_info
    }

    fn get_active_tracks(&self) -> Vec<ActiveTrackInfo<T>> {
        self.tracklets
            .iter()
            .map(|t| ActiveTrackInfo {
                info: TrackInfo {
                    uuid: t.id,
                    tracked_location: t.get_predicted_location(),
                    count: t.count,
                    created: t.created,
                    last_updated: t.last_updated,
                },
                last_box: t.last_box.clone(),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    #[test]
    fn test_vaalbox_xyah_roundtrip() {
        let box1 = [0.0134, 0.02135, 0.12438, 0.691];
        let xyah = xyxy_to_xyah(&box1);
        let box2 = xyah_to_xyxy(&xyah);

        assert!((box1[0] - box2[0]).abs() < f32::EPSILON);
        assert!((box1[1] - box2[1]).abs() < f32::EPSILON);
        assert!((box1[2] - box2[2]).abs() < f32::EPSILON);
        assert!((box1[3] - box2[3]).abs() < f32::EPSILON);
    }

    #[test]
    fn test_iou_identical_boxes() {
        let box1 = [0.1, 0.1, 0.5, 0.5];
        let box2 = [0.1, 0.1, 0.5, 0.5];
        let result = iou(&box1, &box2);
        assert!(
            (result - 1.0).abs() < 0.001,
            "IOU of identical boxes should be 1.0"
        );
    }

    #[test]
    fn test_iou_no_overlap() {
        let box1 = [0.0, 0.0, 0.2, 0.2];
        let box2 = [0.5, 0.5, 0.7, 0.7];
        let result = iou(&box1, &box2);
        assert!(result < 0.001, "IOU of non-overlapping boxes should be ~0");
    }

    #[test]
    fn test_iou_partial_overlap() {
        let box1 = [0.0, 0.0, 0.5, 0.5];
        let box2 = [0.25, 0.25, 0.75, 0.75];
        let result = iou(&box1, &box2);
        // Intersection: 0.25*0.25 = 0.0625, Union: 0.25+0.25-0.0625 = 0.4375
        assert!(result > 0.1 && result < 0.2, "IOU should be ~0.14");
    }

    #[test]
    fn test_bytetrack_new() {
        let tracker: ByteTrack<MockDetection> = ByteTrackBuilder::new().build();
        assert_eq!(tracker.frame_count, 0);
        assert!(tracker.tracklets.is_empty());
        assert_eq!(tracker.track_high_conf, 0.7);
        assert_eq!(tracker.track_iou, 0.25);
    }

    #[test]
    fn test_bytetrack_single_detection_creates_tracklet() {
        let mut tracker = ByteTrackBuilder::new().build();
        let detections = vec![MockDetection::new([0.1, 0.1, 0.3, 0.3], 0.9, 0)];

        let results = tracker.update(&detections, 1000);

        assert_eq!(results.len(), 1);
        assert!(
            results[0].is_some(),
            "High-confidence detection should create tracklet"
        );
        assert_eq!(tracker.tracklets.len(), 1);
        assert_eq!(tracker.frame_count, 1);
    }

    #[test]
    fn test_bytetrack_low_confidence_no_tracklet() {
        let mut tracker = ByteTrackBuilder::new().build();
        // Score below track_high_conf (0.7)
        let detections = vec![MockDetection::new([0.1, 0.1, 0.3, 0.3], 0.5, 0)];

        let results = tracker.update(&detections, 1000);

        assert_eq!(results.len(), 1);
        assert!(
            results[0].is_none(),
            "Low-confidence detection should not create tracklet"
        );
        assert!(tracker.tracklets.is_empty());
    }

    #[test]
    fn test_bytetrack_tracking_across_frames() {
        let mut tracker = ByteTrackBuilder::new().build();

        // Frame 1: Create tracklet with a larger box that's easier to track
        let det1 = vec![MockDetection::new([0.2, 0.2, 0.4, 0.4], 0.9, 0)];
        let res1 = tracker.update(&det1, 1000);
        assert!(res1[0].is_some());
        let uuid1 = res1[0].unwrap().uuid;
        assert_eq!(tracker.tracklets.len(), 1);
        // After creation, tracklet count is 1
        assert_eq!(tracker.tracklets[0].count, 1);

        // Frame 2: Same location - should match existing tracklet
        let det2 = vec![MockDetection::new([0.2, 0.2, 0.4, 0.4], 0.9, 0)];
        let res2 = tracker.update(&det2, 2000);
        assert!(res2[0].is_some());
        let info2 = res2[0].unwrap();

        // Verify tracklet was matched, not a new one created
        assert_eq!(tracker.tracklets.len(), 1, "Should still have one tracklet");
        assert_eq!(info2.uuid, uuid1, "Should match same tracklet");
        // After second update, the internal tracklet count should be 2
        assert_eq!(tracker.tracklets[0].count, 2, "Internal count should be 2");
    }

    #[test]
    fn test_bytetrack_multiple_detections() {
        let mut tracker = ByteTrackBuilder::new().build();

        let detections = vec![
            MockDetection::new([0.1, 0.1, 0.2, 0.2], 0.9, 0),
            MockDetection::new([0.5, 0.5, 0.6, 0.6], 0.85, 0),
            MockDetection::new([0.8, 0.8, 0.9, 0.9], 0.95, 0),
        ];

        let results = tracker.update(&detections, 1000);

        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.is_some()));
        assert_eq!(tracker.tracklets.len(), 3);
    }

    #[test]
    fn test_bytetrack_tracklet_expiry() {
        let mut tracker = ByteTrackBuilder::new().build();
        tracker.track_extra_lifespan = 1000; // 1 second

        // Create tracklet
        let det1 = vec![MockDetection::new([0.1, 0.1, 0.3, 0.3], 0.9, 0)];
        tracker.update(&det1, 1000);
        assert_eq!(tracker.tracklets.len(), 1);

        // Update with no detections after lifespan expires
        let empty: Vec<MockDetection> = vec![];
        tracker.update(&empty, 3000); // 2 seconds later

        assert!(tracker.tracklets.is_empty(), "Tracklet should have expired");
    }

    #[test]
    fn test_bytetrack_get_active_tracks() {
        let mut tracker = ByteTrackBuilder::new().build();

        let detections = vec![
            MockDetection::new([0.1, 0.1, 0.2, 0.2], 0.9, 0),
            MockDetection::new([0.5, 0.5, 0.6, 0.6], 0.85, 0),
        ];
        tracker.update(&detections, 1000);

        let active = tracker.get_active_tracks();
        assert_eq!(active.len(), 2);
        assert!(active.iter().all(|t| t.info.count == 1));
        assert!(active.iter().all(|t| t.info.created == 1000));
    }

    #[test]
    fn test_bytetrack_empty_detections() {
        let mut tracker = ByteTrackBuilder::new().build();
        let empty: Vec<MockDetection> = vec![];

        let results = tracker.update(&empty, 1000);

        assert!(results.is_empty());
        assert!(tracker.tracklets.is_empty());
        assert_eq!(tracker.frame_count, 1);
    }

    #[test]
    fn test_two_stage_matching() {
        // The core ByteTrack innovation: low-confidence detections are matched
        // to existing tracklets in a second stage.
        let mut tracker = ByteTrackBuilder::new().build();

        // Frame 1: high-confidence detection creates a tracklet
        let det1 = vec![MockDetection::new([0.2, 0.2, 0.4, 0.4], 0.9, 0)];
        let res1 = tracker.update(&det1, 1_000_000);
        assert!(res1[0].is_some());
        let uuid1 = res1[0].unwrap().uuid;
        assert_eq!(tracker.tracklets.len(), 1);

        // Frame 2: same location but low confidence (0.3, below track_high_conf=0.7).
        // Second-stage matching should still associate it with the existing tracklet.
        let det2 = vec![MockDetection::new([0.2, 0.2, 0.4, 0.4], 0.3, 0)];
        let res2 = tracker.update(&det2, 2_000_000);
        assert!(
            res2[0].is_some(),
            "Low-conf detection should match existing tracklet via second stage"
        );
        assert_eq!(
            res2[0].unwrap().uuid,
            uuid1,
            "Should match the same tracklet"
        );
        assert_eq!(
            tracker.tracklets.len(),
            1,
            "No new tracklet should be created"
        );
        assert_eq!(
            tracker.tracklets[0].count, 2,
            "Tracklet count should increment"
        );
    }

    #[test]
    fn test_builder_track_extra_lifespan() {
        let lifespan_default = 500_000_000; // 0.5 seconds (default)
        let lifespan_extended = 2_000_000_000; // 2 seconds

        let mut tracker_default: ByteTrack<MockDetection> = ByteTrackBuilder::new().build();
        let mut tracker_extended: ByteTrack<MockDetection> = ByteTrackBuilder::new()
            .track_extra_lifespan(lifespan_extended)
            .build();

        assert_eq!(tracker_default.track_extra_lifespan, lifespan_default);
        assert_eq!(tracker_extended.track_extra_lifespan, lifespan_extended);

        let ts_start = 1_000_000_000u64; // 1 second
        let det = vec![MockDetection::new([0.2, 0.2, 0.4, 0.4], 0.9, 0)];

        tracker_default.update(&det, ts_start);
        tracker_extended.update(&det, ts_start);
        assert_eq!(tracker_default.tracklets.len(), 1);
        assert_eq!(tracker_extended.tracklets.len(), 1);

        // Advance to 1s + 1s = 2s. Default lifespan (0.5s) should have expired,
        // extended lifespan (2s) should still be active.
        let ts_after = ts_start + 1_000_000_000;
        let empty: Vec<MockDetection> = vec![];
        tracker_default.update(&empty, ts_after);
        tracker_extended.update(&empty, ts_after);

        assert!(
            tracker_default.tracklets.is_empty(),
            "Default tracker should have expired the tracklet"
        );
        assert_eq!(
            tracker_extended.tracklets.len(),
            1,
            "Extended tracker should still have the tracklet"
        );
    }

    #[test]
    fn test_builder_track_high_conf() {
        let mut tracker: ByteTrack<MockDetection> =
            ByteTrackBuilder::new().track_high_conf(0.9).build();
        assert_eq!(tracker.track_high_conf, 0.9);

        // Detection with score 0.8 is below the 0.9 threshold
        let det_low = vec![MockDetection::new([0.1, 0.1, 0.3, 0.3], 0.8, 0)];
        let res = tracker.update(&det_low, 1000);
        assert!(
            res[0].is_none(),
            "Score 0.8 should not create a tracklet with threshold 0.9"
        );
        assert!(tracker.tracklets.is_empty());

        // Detection with score 0.95 is above the 0.9 threshold
        let det_high = vec![MockDetection::new([0.1, 0.1, 0.3, 0.3], 0.95, 0)];
        let res = tracker.update(&det_high, 2000);
        assert!(
            res[0].is_some(),
            "Score 0.95 should create a tracklet with threshold 0.9"
        );
        assert_eq!(tracker.tracklets.len(), 1);
    }

    #[test]
    fn test_builder_track_iou() {
        // Tight IOU threshold: shifted detection should NOT match
        let mut tracker: ByteTrack<MockDetection> = ByteTrackBuilder::new().track_iou(0.8).build();

        // Frame 1: two well-separated detections
        let det1 = vec![
            MockDetection::new([0.1, 0.1, 0.3, 0.3], 0.9, 0),
            MockDetection::new([0.5, 0.5, 0.7, 0.7], 0.9, 0),
        ];
        tracker.update(&det1, 1000);
        assert_eq!(tracker.tracklets.len(), 2);

        // Frame 2: shift the first detection slightly. With IOU threshold 0.8
        // the overlap won't be enough for a match, so it creates a new tracklet.
        let det2 = vec![
            MockDetection::new([0.15, 0.15, 0.35, 0.35], 0.9, 0),
            MockDetection::new([0.5, 0.5, 0.7, 0.7], 0.9, 0),
        ];
        let res2 = tracker.update(&det2, 2000);
        assert_eq!(res2.len(), 2);

        // The second detection (unchanged) should still match. The first (shifted)
        // should fail the tight IOU threshold and create a new tracklet.
        assert!(
            tracker.tracklets.len() >= 3,
            "Shifted detection should create a new tracklet with tight IOU threshold, got {} tracklets",
            tracker.tracklets.len()
        );
    }

    #[test]
    fn test_degenerate_zero_area_box() {
        // A zero-area box (xmin == xmax) should not panic
        let mut tracker = ByteTrackBuilder::new().build();
        let det = vec![
            MockDetection::new([0.5, 0.1, 0.5, 0.3], 0.9, 0), // zero width
            MockDetection::new([0.1, 0.1, 0.3, 0.3], 0.9, 0), // normal box
        ];
        let results = tracker.update(&det, 1000);
        assert_eq!(results.len(), 2);

        // IOU between a zero-area box and a normal box should be 0
        let zero_box = [0.5, 0.1, 0.5, 0.3];
        let normal_box = [0.1, 0.1, 0.3, 0.3];
        let iou_val = iou(&zero_box, &normal_box);
        assert!(
            iou_val < EPSILON,
            "IOU with a zero-area box should be ~0, got {iou_val}"
        );
    }

    #[test]
    fn test_degenerate_high_velocity() {
        let mut tracker = ByteTrackBuilder::new().build();

        // Frame 1: detection at top-left
        let det1 = vec![MockDetection::new([0.1, 0.1, 0.2, 0.2], 0.9, 0)];
        let res1 = tracker.update(&det1, 1_000_000);
        assert!(res1[0].is_some());
        let uuid1 = res1[0].unwrap().uuid;
        assert_eq!(tracker.tracklets.len(), 1);

        // Frame 2: detection at bottom-right (huge displacement)
        let det2 = vec![MockDetection::new([0.8, 0.8, 0.9, 0.9], 0.9, 0)];
        let res2 = tracker.update(&det2, 2_000_000);
        assert!(res2[0].is_some());

        // With default IOU threshold the far-away detection should not match;
        // a new tracklet is created instead.
        assert_eq!(
            tracker.tracklets.len(),
            2,
            "Far-displaced detection should create a new tracklet"
        );
        assert_ne!(
            res2[0].unwrap().uuid,
            uuid1,
            "New detection should have a different UUID"
        );
    }

    #[test]
    fn test_many_detections_100() {
        let mut tracker = ByteTrackBuilder::new().build();

        // Generate 100 non-overlapping small boxes spread across [0, 1]
        let detections: Vec<MockDetection> = (0..100)
            .map(|i| {
                let x = (i % 10) as f32 * 0.1;
                let y = (i / 10) as f32 * 0.1;
                MockDetection::new([x, y, x + 0.05, y + 0.05], 0.9, 0)
            })
            .collect();

        let results = tracker.update(&detections, 1000);
        assert_eq!(results.len(), 100);
        assert!(
            results.iter().all(|r| r.is_some()),
            "All 100 high-confidence detections should create tracklets"
        );
        assert_eq!(
            tracker.tracklets.len(),
            100,
            "Should have 100 active tracklets"
        );
    }

    #[test]
    fn test_tracklet_count_increments_each_frame() {
        let mut tracker = ByteTrackBuilder::new().build();
        let det = vec![MockDetection::new([0.2, 0.2, 0.4, 0.4], 0.9, 0)];

        for frame in 1..=5 {
            tracker.update(&det, frame * 1000);
        }

        assert_eq!(tracker.tracklets.len(), 1);
        assert_eq!(
            tracker.tracklets[0].count, 5,
            "Tracklet count should equal number of frames it was matched"
        );
    }

    #[test]
    fn test_tracklet_created_timestamp_preserved() {
        let mut tracker = ByteTrackBuilder::new().build();
        let det = vec![MockDetection::new([0.2, 0.2, 0.4, 0.4], 0.9, 0)];

        tracker.update(&det, 1000);
        tracker.update(&det, 2000);
        tracker.update(&det, 3000);

        let active = tracker.get_active_tracks();
        assert_eq!(active.len(), 1);
        assert_eq!(
            active[0].info.created, 1000,
            "Created timestamp should remain at the first frame"
        );
        assert_eq!(
            active[0].info.last_updated, 3000,
            "Last updated should be the most recent frame"
        );
    }

    #[test]
    fn test_mixed_confidence_detections() {
        // Mix of high and low confidence detections in a single frame
        let mut tracker = ByteTrackBuilder::new().build();
        let det = vec![
            MockDetection::new([0.1, 0.1, 0.2, 0.2], 0.9, 0), // high
            MockDetection::new([0.3, 0.3, 0.4, 0.4], 0.3, 0), // low
            MockDetection::new([0.5, 0.5, 0.6, 0.6], 0.85, 0), // high
            MockDetection::new([0.7, 0.7, 0.8, 0.8], 0.1, 0), // low
        ];

        let results = tracker.update(&det, 1000);
        assert_eq!(results.len(), 4);

        // Only the high-confidence ones should create tracklets
        assert!(
            results[0].is_some(),
            "High-conf detection should create tracklet"
        );
        assert!(
            results[1].is_none(),
            "Low-conf detection should not create tracklet"
        );
        assert!(
            results[2].is_some(),
            "High-conf detection should create tracklet"
        );
        assert!(
            results[3].is_none(),
            "Low-conf detection should not create tracklet"
        );
        assert_eq!(tracker.tracklets.len(), 2);
    }

    #[test]
    fn test_iou_contained_box() {
        // One box fully contains the other
        let outer = [0.0, 0.0, 1.0, 1.0];
        let inner = [0.25, 0.25, 0.75, 0.75];
        let result = iou(&outer, &inner);
        // inner area = 0.25, outer area = 1.0, intersection = 0.25, union = 1.0
        assert!(
            (result - 0.25).abs() < 0.01,
            "IOU of contained box should be inner_area/outer_area = 0.25, got {result}"
        );
    }

    #[test]
    fn test_xyxy_to_xyah_square_box() {
        // A square box should have aspect ratio 1.0
        let square = [0.1, 0.2, 0.3, 0.4];
        let xyah = xyxy_to_xyah(&square);
        assert!((xyah[0] - 0.2).abs() < 1e-5, "Center x should be 0.2");
        assert!((xyah[1] - 0.3).abs() < 1e-5, "Center y should be 0.3");
        assert!(
            (xyah[2] - 1.0).abs() < 1e-5,
            "Aspect ratio of square should be 1.0"
        );
        assert!((xyah[3] - 0.2).abs() < 1e-5, "Height should be 0.2");
    }

    #[test]
    fn test_frame_count_increments() {
        let mut tracker = ByteTrackBuilder::new().build();
        let empty: Vec<MockDetection> = vec![];

        for _ in 0..10 {
            tracker.update(&empty, 0);
        }

        assert_eq!(
            tracker.frame_count, 10,
            "Frame count should increment each update"
        );
    }

    #[test]
    fn test_tracklet_predicted_location_near_detection() {
        let mut tracker = ByteTrackBuilder::new().build();
        let det = vec![MockDetection::new([0.2, 0.2, 0.4, 0.4], 0.9, 0)];
        tracker.update(&det, 1000);

        let pred = tracker.tracklets[0].get_predicted_location();
        // The predicted location should be close to the original detection
        assert!(
            (pred[0] - 0.2).abs() < 0.1,
            "Predicted xmin should be near 0.2, got {}",
            pred[0]
        );
        assert!(
            (pred[1] - 0.2).abs() < 0.1,
            "Predicted ymin should be near 0.2, got {}",
            pred[1]
        );
        assert!(
            (pred[2] - 0.4).abs() < 0.1,
            "Predicted xmax should be near 0.4, got {}",
            pred[2]
        );
        assert!(
            (pred[3] - 0.4).abs() < 0.1,
            "Predicted ymax should be near 0.4, got {}",
            pred[3]
        );
    }
}
