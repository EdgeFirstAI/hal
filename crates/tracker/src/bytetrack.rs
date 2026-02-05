// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::{kalman::ConstantVelocityXYAHModel2, DetectionBox, TrackInfo, Tracker};
use lapjv::{lapjv, Matrix};
use log::{debug, trace};
use nalgebra::{Dyn, OMatrix, U4};
use uuid::Uuid;

#[allow(dead_code)]
#[derive(Default)]
pub struct ByteTrack {
    pub track_extra_lifespan: u64,
    pub track_high_conf: f32,
    pub track_iou: f32,
    pub track_update: f32,

    pub tracklets: Vec<Tracklet>,
    pub frame_count: i32,
}

#[derive(Debug, Clone)]
pub struct Tracklet {
    pub id: Uuid,
    pub filter: ConstantVelocityXYAHModel2<f32>,
    pub count: i32,
    pub created: u64,
    pub last_updated: u64,
}

impl Tracklet {
    fn update<T: DetectionBox>(&mut self, detect_box: &T, ts: u64) {
        self.count += 1;
        self.last_updated = ts;
        self.filter.update(&vaalbox_to_xyah(&detect_box.bbox()));
    }

    pub fn get_predicted_location(&self) -> [f32; 4] {
        let predicted_xyah = self.filter.mean.as_slice();
        xyah_to_vaalbox(predicted_xyah)
    }
}

fn vaalbox_to_xyah(vaal_box: &[f32; 4]) -> [f32; 4] {
    let x = (vaal_box[2] + vaal_box[0]) / 2.0;
    let y = (vaal_box[3] + vaal_box[1]) / 2.0;
    let w = (vaal_box[2] - vaal_box[0]).max(EPSILON);
    let h = (vaal_box[3] - vaal_box[1]).max(EPSILON);
    let a = w / h;

    [x, y, a, h]
}

fn xyah_to_vaalbox(xyah: &[f32]) -> [f32; 4] {
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
    track: &Tracklet,
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
    let expected = xyah_to_vaalbox(predicted_xyah);
    let iou = iou(&expected, &new_box.bbox());
    if iou < iou_threshold {
        return INVALID_MATCH;
    }
    (1.5 - new_box.score()) + (1.5 - iou)
}

impl ByteTrack {
    pub fn new() -> ByteTrack {
        ByteTrack {
            track_extra_lifespan: 500_000_000,
            track_high_conf: 0.7,
            track_iou: 0.25,
            track_update: 0.25,
            tracklets: Vec::new(),
            frame_count: 0,
        }
    }

    fn compute_costs<T: DetectionBox>(
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
            row.copy_from_slice(&vaalbox_to_xyah(&boxes[i].bbox()));
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
    fn process_assignments<T: DetectionBox>(
        &mut self,
        assignments: &[usize],
        boxes: &[T],
        costs: &Matrix<f32>,
        matched: &mut [bool],
        tracked: &mut [bool],
        matched_info: &mut [Option<TrackInfo>],
        timestamp: u64,
        skip_already_matched: bool,
    ) {
        for (i, &x) in assignments.iter().enumerate() {
            if i >= boxes.len() || x >= self.tracklets.len() {
                continue;
            }

            // Filter out invalid assignments
            if costs[(i, x)] >= INVALID_MATCH {
                continue;
            }

            // For second pass, skip already matched boxes/tracklets
            if skip_already_matched && (matched[i] || tracked[x]) {
                continue;
            }

            if skip_already_matched {
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
            assert!(!tracked[x]);
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
                debug!("Tracklet removed: {:?}", self.tracklets[i].id);
                let _ = self.tracklets.swap_remove(i);
            }
        }
    }

    /// Create new tracklets from unmatched high-confidence boxes.
    fn create_new_tracklets<T: DetectionBox>(
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
                    &vaalbox_to_xyah(&boxes[i].bbox()),
                    self.track_update,
                ),
                last_updated: timestamp,
                count: 1,
                created: timestamp,
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

impl<T> Tracker<T> for ByteTrack
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
            let ans = lapjv(&costs).unwrap();
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

        // Second pass: match remaining tracklets to low-confidence detections
        if !self.tracklets.is_empty() {
            let costs = self.compute_costs(boxes, 0.0, self.track_iou, &matched, &tracked);
            let ans = lapjv(&costs).unwrap();
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

    fn get_active_tracks(&self) -> Vec<TrackInfo> {
        self.tracklets
            .iter()
            .map(|t| TrackInfo {
                uuid: t.id,
                tracked_location: t.get_predicted_location(),
                count: t.count,
                created: t.created,
                last_updated: t.last_updated,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{iou, vaalbox_to_xyah, xyah_to_vaalbox, ByteTrack};
    use crate::{DetectionBox, Tracker};

    /// Mock detection for testing
    #[derive(Debug, Clone)]
    struct MockDetection {
        bbox: [f32; 4],
        score: f32,
        label: usize,
    }

    impl MockDetection {
        fn new(x1: f32, y1: f32, x2: f32, y2: f32, score: f32) -> Self {
            Self {
                bbox: [x1, y1, x2, y2],
                score,
                label: 0,
            }
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

    #[test]
    fn test_vaalbox_xyah_roundtrip() {
        let box1 = [0.0134, 0.02135, 0.12438, 0.691];
        let xyah = vaalbox_to_xyah(&box1);
        let box2 = xyah_to_vaalbox(&xyah);

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
        let tracker = ByteTrack::new();
        assert_eq!(tracker.frame_count, 0);
        assert!(tracker.tracklets.is_empty());
        assert_eq!(tracker.track_high_conf, 0.7);
        assert_eq!(tracker.track_iou, 0.25);
    }

    #[test]
    fn test_bytetrack_single_detection_creates_tracklet() {
        let mut tracker = ByteTrack::new();
        let detections = vec![MockDetection::new(0.1, 0.1, 0.3, 0.3, 0.9)];

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
        let mut tracker = ByteTrack::new();
        // Score below track_high_conf (0.7)
        let detections = vec![MockDetection::new(0.1, 0.1, 0.3, 0.3, 0.5)];

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
        let mut tracker = ByteTrack::new();

        // Frame 1: Create tracklet with a larger box that's easier to track
        let det1 = vec![MockDetection::new(0.2, 0.2, 0.4, 0.4, 0.9)];
        let res1 = tracker.update(&det1, 1000);
        assert!(res1[0].is_some());
        let uuid1 = res1[0].unwrap().uuid;
        assert_eq!(tracker.tracklets.len(), 1);
        // After creation, tracklet count is 1
        assert_eq!(tracker.tracklets[0].count, 1);

        // Frame 2: Same location - should match existing tracklet
        let det2 = vec![MockDetection::new(0.2, 0.2, 0.4, 0.4, 0.9)];
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
        let mut tracker = ByteTrack::new();

        let detections = vec![
            MockDetection::new(0.1, 0.1, 0.2, 0.2, 0.9),
            MockDetection::new(0.5, 0.5, 0.6, 0.6, 0.85),
            MockDetection::new(0.8, 0.8, 0.9, 0.9, 0.95),
        ];

        let results = tracker.update(&detections, 1000);

        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.is_some()));
        assert_eq!(tracker.tracklets.len(), 3);
    }

    #[test]
    fn test_bytetrack_tracklet_expiry() {
        let mut tracker = ByteTrack::new();
        tracker.track_extra_lifespan = 1000; // 1 second

        // Create tracklet
        let det1 = vec![MockDetection::new(0.1, 0.1, 0.3, 0.3, 0.9)];
        tracker.update(&det1, 1000);
        assert_eq!(tracker.tracklets.len(), 1);

        // Update with no detections after lifespan expires
        let empty: Vec<MockDetection> = vec![];
        tracker.update(&empty, 3000); // 2 seconds later

        assert!(tracker.tracklets.is_empty(), "Tracklet should have expired");
    }

    #[test]
    fn test_bytetrack_get_active_tracks() {
        let mut tracker = ByteTrack::new();

        let detections = vec![
            MockDetection::new(0.1, 0.1, 0.2, 0.2, 0.9),
            MockDetection::new(0.5, 0.5, 0.6, 0.6, 0.85),
        ];
        tracker.update(&detections, 1000);

        let active = <ByteTrack as Tracker<MockDetection>>::get_active_tracks(&tracker);
        assert_eq!(active.len(), 2);
        assert!(active.iter().all(|t| t.count == 1));
        assert!(active.iter().all(|t| t.created == 1000));
    }

    #[test]
    fn test_bytetrack_empty_detections() {
        let mut tracker = ByteTrack::new();
        let empty: Vec<MockDetection> = vec![];

        let results = tracker.update(&empty, 1000);

        assert!(results.is_empty());
        assert!(tracker.tracklets.is_empty());
        assert_eq!(tracker.frame_count, 1);
    }
}
