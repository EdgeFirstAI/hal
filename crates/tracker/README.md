# edgefirst-tracker

[![Crates.io](https://img.shields.io/crates/v/edgefirst-tracker.svg)](https://crates.io/crates/edgefirst-tracker)
[![Documentation](https://docs.rs/edgefirst-tracker/badge.svg)](https://docs.rs/edgefirst-tracker)
[![License](https://img.shields.io/crates/l/edgefirst-tracker.svg)](LICENSE)

**Multi-object tracking for edge AI applications.**

This crate provides object tracking algorithms for associating detections across video frames, enabling applications like counting, path analysis, and re-identification.

## Algorithms

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **ByteTrack** | High-performance MOT with two-pass association and Kalman filtering | Real-time tracking |

## Features

- **UUID-based track IDs** - Globally unique identifiers for each tracked object
- **Kalman filtering** - Smooth trajectory prediction and state estimation
- **Two-pass association** - High-confidence detections matched first, low-confidence detections used to recover lost tracks
- **Configurable thresholds** - Tunable IoU, confidence, and lifespan parameters
- **Track lifecycle** - Automatic creation, update, and expiry of tracks

## Quick Start

```rust,ignore
use edgefirst_tracker::{bytetrack::{ByteTrack, ByteTrackBuilder}, Tracker, TrackInfo};

// Build a tracker using the builder pattern
let mut tracker: ByteTrack<MyDetection> = ByteTrackBuilder::new()
    .track_high_conf(0.7)        // minimum confidence to create a new track
    .track_iou(0.25)             // IoU threshold for matching
    .track_update(0.25)          // Kalman filter update rate
    .track_extra_lifespan(500_000_000)  // nanoseconds to keep a track alive without a match
    .build();

// Process detections each frame
let detections: Vec<MyDetection> = get_detections_from_model();
let timestamp: u64 = frame_timestamp_ns;

let tracks: Vec<Option<TrackInfo>> = tracker.update(&detections, timestamp);

// Each detection gets associated track info (or None if untracked)
for (det, track) in detections.iter().zip(tracks.iter()) {
    if let Some(info) = track {
        println!("Detection matched to track {}", info.uuid);
        println!("  Track age: {} updates", info.count);
        println!("  Kalman-smoothed location: {:?}", info.tracked_location);
    }
}

// Query all currently active tracks
for active in tracker.get_active_tracks() {
    println!(
        "Active track {} — location {:?}, last raw box: {:?}",
        active.info.uuid, active.info.tracked_location, active.last_box
    );
}
```

## Default Parameters

`ByteTrackBuilder::new()` sets the following defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `track_high_conf` | `0.7` | Minimum score to create or match a high-confidence track |
| `track_iou` | `0.25` | IoU threshold for box association |
| `track_update` | `0.25` | Kalman filter update rate |
| `track_extra_lifespan` | `500_000_000` ns | How long a track survives without a matching detection |

## Track Information

`Tracker::update()` returns `Vec<Option<TrackInfo>>`, one entry per input detection. Detections below the high-confidence threshold that do not match an existing track return `None`.

Each `TrackInfo` contains:

| Field | Type | Description |
|-------|------|-------------|
| `uuid` | `Uuid` | Globally unique track identifier |
| `tracked_location` | `[f32; 4]` | Kalman-filtered bounding box in XYXY format `[xmin, ymin, xmax, ymax]` |
| `count` | `i32` | Number of times this track has been updated |
| `created` | `u64` | Timestamp (ns) when the track was first created |
| `last_updated` | `u64` | Timestamp (ns) of the last detection match |

`Tracker::get_active_tracks()` returns `Vec<ActiveTrackInfo<T>>`, which bundles `TrackInfo` with the raw last matched detection box:

| Field | Type | Description |
|-------|------|-------------|
| `info` | `TrackInfo` | Tracking metadata (see above) |
| `last_box` | `T` | The last raw detection box before Kalman smoothing |

## Implementing DetectionBox

Any type used with `ByteTrack<T>` must implement the `DetectionBox` trait:

```rust
use edgefirst_tracker::DetectionBox;

#[derive(Debug, Clone)]
struct MyDetection {
    bbox: [f32; 4],
    confidence: f32,
    class_id: usize,
}

impl DetectionBox for MyDetection {
    fn bbox(&self) -> [f32; 4] { self.bbox }         // XYXY format
    fn score(&self) -> f32 { self.confidence }
    fn label(&self) -> usize { self.class_id }
}
```

The `bbox()` method must return coordinates in XYXY format: `[xmin, ymin, xmax, ymax]` as normalized or pixel values consistent with the model output.

## The Tracker Trait

Both `ByteTrack<T>` and any custom tracker must implement:

```rust
pub trait Tracker<T: DetectionBox> {
    fn update(&mut self, boxes: &[T], timestamp: u64) -> Vec<Option<TrackInfo>>;
    fn get_active_tracks(&self) -> Vec<ActiveTrackInfo<T>>;
}
```

- `timestamp` is expected in nanoseconds and is used for track expiry calculations.
- The returned `Vec` from `update` is aligned 1-to-1 with the input `boxes` slice.

## Integration with edgefirst-decoder

When used via `edgefirst-hal` or `edgefirst-decoder`, the `tracker` feature flag must be enabled:

```toml
[dependencies]
edgefirst-hal = { version = "0.13", features = ["tracker"] }
```

Or when depending on the decoder directly:

```toml
[dependencies]
edgefirst-decoder = { version = "0.13", features = ["tracker"] }
```

The decoder exposes `decode_tracked()` which accepts any `Tracker<DetectBox>` implementation:

```rust,ignore
use edgefirst_tracker::{bytetrack::ByteTrackBuilder, Tracker};

let mut tracker = ByteTrackBuilder::new()
    .track_high_conf(0.8)
    .build();

decoder.decode_tracked(
    &mut tracker,
    timestamp_ns,
    &tensor_outputs,
    &mut output_boxes,
    &mut output_masks,
    &mut output_tracks,
)?;
```

`decode_tracked()` populates:
- `output_boxes` — decoded detection boxes
- `output_masks` — segmentation masks (empty if the model has no mask head)
- `output_tracks` — `Vec<TrackInfo>` with one entry per matched detection

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/EdgeFirstAI/hal/blob/main/LICENSE) for details.
