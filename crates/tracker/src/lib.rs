// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use std::{
    cell::RefCell,
    fmt::Debug,
    rc::Rc,
    sync::{Arc, Mutex},
};

pub use uuid::Uuid;
pub mod bytetrack;
pub use bytetrack::{ByteTrack, ByteTrackBuilder};
mod kalman;

pub trait DetectionBox: Debug + Clone {
    fn bbox(&self) -> [f32; 4];
    fn score(&self) -> f32;
    fn label(&self) -> usize;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackInfo {
    pub uuid: Uuid,
    pub tracked_location: [f32; 4],
    pub count: i32,
    pub created: u64,
    pub last_updated: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActiveTrackInfo<T: DetectionBox> {
    pub info: TrackInfo,
    // /// The last raw box associated with the track. This is before any smoothing
    // /// introduced by the tracker
    pub last_box: T,
}

pub trait Tracker<T: DetectionBox> {
    fn update(&mut self, boxes: &[T], timestamp: u64) -> Vec<Option<TrackInfo>>;
    fn get_active_tracks(&self) -> Vec<ActiveTrackInfo<T>>;
}

impl<T, TR> Tracker<T> for Arc<Mutex<TR>>
where
    T: DetectionBox,
    TR: Tracker<T>,
{
    fn update(&mut self, boxes: &[T], timestamp: u64) -> Vec<Option<TrackInfo>> {
        let mut tracker = self.lock().unwrap();
        tracker.update(boxes, timestamp)
    }

    fn get_active_tracks(&self) -> Vec<ActiveTrackInfo<T>> {
        let tracker = self.lock().unwrap();
        tracker.get_active_tracks()
    }
}

impl<T, TR> Tracker<T> for Rc<RefCell<TR>>
where
    T: DetectionBox,
    TR: Tracker<T>,
{
    fn update(&mut self, boxes: &[T], timestamp: u64) -> Vec<Option<TrackInfo>> {
        let mut tracker = self.borrow_mut();
        tracker.update(boxes, timestamp)
    }
    fn get_active_tracks(&self) -> Vec<ActiveTrackInfo<T>> {
        let tracker = self.borrow();
        tracker.get_active_tracks()
    }
}
