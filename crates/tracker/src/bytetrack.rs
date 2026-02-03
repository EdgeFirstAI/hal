// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::{DetectionBox, TrackInfo, Tracker, kalman::ConstantVelocityXYAHModel2};
use lapjv::{Matrix, lapjv};
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

    use super::{vaalbox_to_xyah, xyah_to_vaalbox};

    #[test]
    fn filter() {
        let box1 = [0.0134, 0.02135, 0.12438, 0.691];
        let xyah = vaalbox_to_xyah(&box1);

        let box2 = xyah_to_vaalbox(&xyah);
        assert!((box1[2] - box2[2]).abs() < f32::EPSILON);
        assert!((box1[3] - box2[3]).abs() < f32::EPSILON);
        assert!((box1[0] - box2[0]).abs() < f32::EPSILON);
        assert!((box1[1] - box2[1]).abs() < f32::EPSILON);
    }
}
