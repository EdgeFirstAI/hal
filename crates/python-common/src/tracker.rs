// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use std::{
    fmt,
    sync::{Arc, Mutex},
};

use edgefirst_tracker::{
    ActiveTrackInfo, ByteTrack, ByteTrackBuilder, DetectionBox, TrackInfo, Tracker,
};
use numpy::{PyArrayLike1, PyArrayLike2};
use pyo3::{exceptions::PyValueError, pyclass, pymethods, PyResult, Python};
use uuid::Uuid;

/// Local box type so this module does not depend on `edgefirst-tensor`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TrackBox {
    bbox: [f32; 4],
    score: f32,
    label: usize,
}

impl DetectionBox for TrackBox {
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

#[pyclass(
    name = "TrackInfo",
    str,
    eq,
    from_py_object,
    module = "edgefirst.tracker"
)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PyTrackInfo {
    pub track: TrackInfo,
}

impl From<TrackInfo> for PyTrackInfo {
    fn from(track: TrackInfo) -> Self {
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
    ) -> PyResult<Self> {
        Ok(Self {
            track: TrackInfo {
                uuid: Uuid::parse_str(&uuid).map_err(|e| PyValueError::new_err(e.to_string()))?,
                tracked_location,
                count,
                created,
                last_updated,
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

    fn __repr__(&self) -> String {
        format!("{:?}", self.track)
    }
}

#[pyclass(
    name = "ActiveTrackInfo",
    str,
    eq,
    from_py_object,
    module = "edgefirst.tracker"
)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PyActiveTrackInfo {
    pub(crate) track: ActiveTrackInfo<TrackBox>,
}

impl From<ActiveTrackInfo<TrackBox>> for PyActiveTrackInfo {
    fn from(track: ActiveTrackInfo<TrackBox>) -> Self {
        Self { track }
    }
}
impl fmt::Display for PyActiveTrackInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.track)
    }
}

#[pymethods]
impl PyActiveTrackInfo {
    #[new]
    pub fn new(
        track_info: PyTrackInfo,
        bbox: [f32; 4],
        score: f32,
        label: usize,
    ) -> PyResult<Self> {
        let detect_box = TrackBox { bbox, score, label };
        Ok(Self {
            track: ActiveTrackInfo {
                info: track_info.track,
                last_box: detect_box,
            },
        })
    }

    #[getter]
    pub fn info(&self) -> PyTrackInfo {
        self.track.info.into()
    }

    #[getter]
    pub fn last_box(&self) -> ([f32; 4], f32, usize) {
        let last_box = self.track.last_box;
        (last_box.bbox, last_box.score, last_box.label)
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.track)
    }
}

#[pyclass(name = "ByteTrack", from_py_object, module = "edgefirst.tracker")]
#[derive(Clone)]
pub struct PyByteTrack {
    pub(crate) tracker: Arc<Mutex<ByteTrack<TrackBox>>>,
}

unsafe impl Send for PyByteTrack {}
unsafe impl Sync for PyByteTrack {}

impl Tracker<TrackBox> for PyByteTrack {
    fn update(&mut self, boxes: &[TrackBox], timestamp_ns: u64) -> Vec<Option<TrackInfo>> {
        let mut tracker = self.tracker.lock().unwrap_or_else(|e| e.into_inner());
        tracker.update(boxes, timestamp_ns)
    }

    fn get_active_tracks(&self) -> Vec<ActiveTrackInfo<TrackBox>> {
        let tracker = self.tracker.lock().unwrap_or_else(|e| e.into_inner());
        tracker.get_active_tracks()
    }
}

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
        py: Python<'_>,
        boxes: PyArrayLike2<f32>,
        scores: PyArrayLike1<f32>,
        labels: PyArrayLike1<usize>,
        timestamp_ns: u64,
    ) -> Vec<Option<PyTrackInfo>> {
        let boxes = boxes.as_array().to_owned();
        let scores = scores.as_array().to_owned();
        let labels = labels.as_array().to_owned();
        let tracker = Arc::clone(&self.tracker);
        py.detach(move || {
            let boxes: Vec<TrackBox> = boxes
                .rows()
                .into_iter()
                .zip(scores.iter())
                .zip(labels.iter())
                .map(|((bbox, score), label)| TrackBox {
                    bbox: [bbox[0], bbox[1], bbox[2], bbox[3]],
                    score: *score,
                    label: *label,
                })
                .collect();
            let mut tracker = tracker.lock().unwrap_or_else(|e| e.into_inner());
            tracker
                .update(&boxes, timestamp_ns)
                .into_iter()
                .map(|t| t.map(|ti| PyTrackInfo { track: ti }))
                .collect()
        })
    }

    pub fn get_active_tracks(&self, py: Python<'_>) -> Vec<PyActiveTrackInfo> {
        let tracker = Arc::clone(&self.tracker);
        py.detach(move || {
            let tracker = tracker.lock().unwrap_or_else(|e| e.into_inner());
            tracker
                .get_active_tracks()
                .into_iter()
                .map(|ti| ti.into())
                .collect()
        })
    }
}
