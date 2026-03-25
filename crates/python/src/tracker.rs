// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use std::{
    fmt,
    sync::{Arc, Mutex},
};

use edgefirst_hal::decoder::{DetectBox, TrackInfo};
use edgefirst_hal::tracker::{ActiveTrackInfo, ByteTrack, ByteTrackBuilder, Tracker};
use numpy::{PyArrayLike1, PyArrayLike2};
use pyo3::{exceptions::PyValueError, pyclass, pymethods, PyResult};
use uuid::Uuid;

#[pyclass(name = "TrackInfo", str, eq)]
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

#[pyclass(name = "ActiveTrackInfo", str, eq)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PyActiveTrackInfo {
    pub track: ActiveTrackInfo<DetectBox>,
}

impl From<ActiveTrackInfo<DetectBox>> for PyActiveTrackInfo {
    fn from(track: ActiveTrackInfo<DetectBox>) -> Self {
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
        let detect_box = DetectBox {
            bbox: bbox.into(),
            score,
            label,
        };
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
        (last_box.bbox.into(), last_box.score, last_box.label)
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.track)
    }
}

#[pyclass(name = "ByteTrack")]
#[derive(Clone)]
pub struct PyByteTrack {
    pub(crate) tracker: Arc<Mutex<ByteTrack<DetectBox>>>,
}

unsafe impl Send for PyByteTrack {}
unsafe impl Sync for PyByteTrack {}

impl Tracker<DetectBox> for PyByteTrack {
    fn update(&mut self, boxes: &[DetectBox], timestamp_ns: u64) -> Vec<Option<TrackInfo>> {
        let mut tracker = self.tracker.lock().unwrap_or_else(|e| e.into_inner());
        tracker.update(boxes, timestamp_ns)
    }

    fn get_active_tracks(&self) -> Vec<ActiveTrackInfo<DetectBox>> {
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

        let tracks = <Self as Tracker<DetectBox>>::update(self, &boxes, timestamp_ns);

        tracks
            .into_iter()
            .map(|t| t.map(|ti| PyTrackInfo { track: ti }))
            .collect()
    }

    pub fn get_active_tracks(&self) -> Vec<PyActiveTrackInfo> {
        let tracks = <Self as Tracker<DetectBox>>::get_active_tracks(self);
        tracks.into_iter().map(|ti| ti.into()).collect()
    }
}
