# edgefirst-tracker

[![Crates.io](https://img.shields.io/crates/v/edgefirst-tracker.svg)](https://crates.io/crates/edgefirst-tracker)
[![Documentation](https://docs.rs/edgefirst-tracker/badge.svg)](https://docs.rs/edgefirst-tracker)
[![License](https://img.shields.io/crates/l/edgefirst-tracker.svg)](LICENSE)

**Multi-object tracking for edge AI applications.**

This crate provides object tracking algorithms for associating detections across video frames, enabling applications like counting, path analysis, and re-identification.

## Algorithms

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **ByteTrack** | High-performance MOT using byte-level features | Real-time tracking |

## Features

- **UUID-based track IDs** - Globally unique identifiers for each tracked object
- **Kalman filtering** - Smooth trajectory prediction and state estimation
- **Configurable association** - IoU-based matching with tunable thresholds
- **Track lifecycle** - Automatic creation, update, and deletion of tracks

## Quick Start

```rust
use edgefirst_tracker::{Tracker, TrackInfo, DetectionBox};
use edgefirst_tracker::bytetrack::ByteTracker;

// Create tracker
let mut tracker = ByteTracker::new(
    30,    // max lost frames before track deletion
    0.5,   // high detection threshold
    0.1,   // low detection threshold
    0.8,   // match threshold
);

// Process detections each frame
let detections = get_detections_from_model();
let timestamp = frame_number as u64;

let tracks = tracker.update(&detections, timestamp);

// Each detection gets associated track info (or None if untracked)
for (det, track) in detections.iter().zip(tracks.iter()) {
    if let Some(info) = track {
        println!("Detection matched to track {}", info.uuid);
        println!("  Track age: {} frames", info.count);
        println!("  Predicted location: {:?}", info.tracked_location);
    }
}

// Get all currently active tracks
for track in tracker.get_active_tracks() {
    println!("Active track: {} at {:?}", track.uuid, track.tracked_location);
}
```

## Track Information

Each `TrackInfo` contains:

| Field | Type | Description |
|-------|------|-------------|
| `uuid` | `Uuid` | Globally unique track identifier |
| `tracked_location` | `[f32; 4]` | Kalman-filtered bounding box `[x1, y1, x2, y2]` |
| `count` | `i32` | Number of frames this track has been active |
| `created` | `u64` | Timestamp when track was created |
| `last_updated` | `u64` | Timestamp of last detection match |

## Integration

Works with any detection source implementing `DetectionBox`:

```rust
use edgefirst_tracker::DetectionBox;

impl DetectionBox for MyDetection {
    fn bbox(&self) -> [f32; 4] { self.coordinates }
    fn score(&self) -> f32 { self.confidence }
    fn label(&self) -> usize { self.class_id }
}
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
