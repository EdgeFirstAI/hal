// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Debug;

use uuid::Uuid;

pub mod bytetrack;
mod kalman;

pub trait DetectionBox: Debug {
    fn bbox(&self) -> [f32; 4];
    fn score(&self) -> f32;
    fn label(&self) -> usize;
}

#[derive(Debug, Clone, Copy)]
pub struct TrackInfo {
    pub uuid: Uuid,
    pub tracked_location: [f32; 4],
    pub count: i32,
    pub created: u64,
    pub last_updated: u64,
}

pub trait Tracker<T: DetectionBox> {
    fn update(&mut self, boxes: &[T], timestamp: u64) -> Vec<Option<TrackInfo>>;

    fn get_active_tracks(&self) -> Vec<TrackInfo>;
}
