// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use std::{
    fmt,
    sync::{Arc, Mutex},
};

use edgefirst::{
    decoder::DetectBox,
    tracker::{ByteTrack, ByteTrackBuilder, TrackInfo, Tracker, Uuid},
};
use numpy::{PyArrayLike1, PyArrayLike2};
use pyo3::{PyResult, exceptions::PyValueError, pyclass, pymethods};

#[pyclass(name = "TrackInfo", str, eq)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PyTrackInfo {
    pub track: TrackInfo<DetectBox>,
}

unsafe impl Send for PyTrackInfo {}
unsafe impl Sync for PyTrackInfo {}

impl From<TrackInfo<DetectBox>> for PyTrackInfo {
    fn from(track: TrackInfo<DetectBox>) -> Self {
        Self { track }
    }
}

impl fmt::Display for PyTrackInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.track)
    }
}

#[pymethods]
impl PyTrackInfo {
    #[new]
    pub fn new(
        uuid: String,
        tracked_location: [f32; 4],
        count: i32,
        created: u64,
        last_updated: u64,
        last_box: ([f32; 4], f32, usize),
    ) -> PyResult<Self> {
        let last_box = DetectBox {
            bbox: last_box.0.into(),
            score: last_box.1,
            label: last_box.2,
        };
        Ok(Self {
            track: TrackInfo {
                uuid: Uuid::parse_str(&uuid).map_err(|e| PyValueError::new_err(e.to_string()))?,
                tracked_location,
                count,
                created,
                last_updated,
                last_box,
            },
        })
    }

    #[getter]
    pub fn uuid(&self) -> String {
        self.track.uuid.to_string()
    }

    #[getter]
    pub fn tracked_location(&self) -> [f32; 4] {
        self.track.tracked_location
    }

    #[getter]
    pub fn count(&self) -> i32 {
        self.track.count
    }

    #[getter]
    pub fn created(&self) -> u64 {
        self.track.created
    }

    #[getter]
    pub fn last_updated(&self) -> u64 {
        self.track.last_updated
    }

    #[getter]
    pub fn last_box(&self) -> ([f32; 4], f32, usize) {
        (
            self.track.last_box.bbox.into(),
            self.track.last_box.score,
            self.track.last_box.label,
        )
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.track)
    }
}

#[pyclass(name = "ByteTrack")]
pub struct PyByteTrack {
    pub(crate) tracker: Arc<Mutex<ByteTrack<DetectBox>>>,
}

unsafe impl Send for PyByteTrack {}
unsafe impl Sync for PyByteTrack {}

#[pymethods]
impl PyByteTrack {
    #[new]
    #[pyo3(signature = (high_conf=0.7, iou=0.25, update=0.25, lifespan_ns=500_000_000))]
    pub fn new(high_conf: f32, iou: f32, update: f32, lifespan_ns: u64) -> Self {
        let update = update.clamp(0.0, 1.0);
        Self {
            tracker: Arc::new(Mutex::new(
                ByteTrackBuilder::new()
                    .track_high_conf(high_conf)
                    .track_iou(iou)
                    .track_update(update)
                    .track_extra_lifespan(lifespan_ns)
                    .build(),
            )),
        }
    }

    pub fn update(
        &mut self,
        boxes: PyArrayLike2<f32>,
        scores: PyArrayLike1<f32>,
        labels: PyArrayLike1<usize>,
        timestamp_ns: u64,
    ) -> Vec<Option<PyTrackInfo>> {
        let boxes = boxes.as_array();
        let scores = scores.as_array();
        let labels = labels.as_array();

        let boxes = boxes
            .rows()
            .into_iter()
            .zip(scores.iter())
            .zip(labels.iter())
            .map(|((bbox, score), label)| DetectBox {
                bbox: [bbox[0], bbox[1], bbox[2], bbox[3]].into(),
                score: *score,
                label: *label,
            })
            .collect::<Vec<_>>();

        let tracks = self.tracker.update(&boxes, timestamp_ns);

        tracks
            .into_iter()
            .map(|t| t.map(|ti| PyTrackInfo { track: ti }))
            .collect()
    }

    pub fn get_active_tracks(&self) -> Vec<PyTrackInfo> {
        let tracks = self.tracker.lock().unwrap().get_active_tracks();
        tracks
            .into_iter()
            .map(|ti| PyTrackInfo { track: ti })
            .collect()
    }
}
