// SPDX-FileCopyrightText: Copyright 2025-2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use edgefirst_bench::{run_bench, BenchSuite};
use edgefirst_tracker::{
    bytetrack::ByteTrackBuilder, kalman::ConstantVelocityXYAHModel2, MockDetection, Tracker,
};

const WARMUP: usize = 10;
const ITERATIONS: usize = 100;

/// Generate non-overlapping detections spread across the [0,1] coordinate space.
fn generate_detections(n: usize) -> Vec<MockDetection> {
    let cols = (n as f32).sqrt().ceil() as usize;
    let size = 0.8 / cols as f32;
    let mut detections = Vec::with_capacity(n);
    for i in 0..n {
        let row = i / cols;
        let col = i % cols;
        let x = col as f32 / cols as f32;
        let y = row as f32 / cols as f32;
        detections.push(MockDetection::new([x, y, x + size, y + size], 0.9, 0));
    }
    detections
}

fn bench_bytetrack_update(suite: &mut BenchSuite, n: usize) {
    let detections = generate_detections(n);
    let name = format!("tracker/bytetrack/update/{n}");

    let result = run_bench(&name, WARMUP, ITERATIONS, || {
        let mut tracker = ByteTrackBuilder::new()
            .track_update(0.25)
            .track_high_conf(0.7)
            .build();
        tracker.update(&detections, 1_000_000_000);
    });
    result.print_summary();
    suite.record(&result);
}

fn bench_bytetrack_100_frames(suite: &mut BenchSuite) {
    let detections = generate_detections(10);

    let result = run_bench("tracker/bytetrack/100_frames", WARMUP, ITERATIONS, || {
        let mut tracker = ByteTrackBuilder::new()
            .track_update(0.25)
            .track_high_conf(0.7)
            .build();
        for frame in 0..100u64 {
            let ts = frame * 33_333_333; // ~30 fps
            tracker.update(&detections, ts);
        }
    });
    result.print_summary();
    suite.record(&result);
}

fn bench_kalman_predict(suite: &mut BenchSuite) {
    let result = run_bench("tracker/kalman/predict", WARMUP, ITERATIONS, || {
        let mut filter = ConstantVelocityXYAHModel2::<f32>::new(&[0.5, 0.5, 1.0, 0.5], 0.25);
        for _ in 0..1000 {
            filter.predict();
        }
    });
    result.print_summary();
    suite.record(&result);
}

fn bench_kalman_update(suite: &mut BenchSuite) {
    let result = run_bench("tracker/kalman/update", WARMUP, ITERATIONS, || {
        let mut filter = ConstantVelocityXYAHModel2::<f32>::new(&[0.5, 0.5, 1.0, 0.5], 0.25);
        filter.predict();
        for _ in 0..1000 {
            filter.update(&[0.51, 0.49, 1.01, 0.50]);
            filter.predict();
        }
    });
    result.print_summary();
    suite.record(&result);
}

fn main() {
    let mut suite = BenchSuite::from_args();

    bench_bytetrack_update(&mut suite, 1);
    bench_bytetrack_update(&mut suite, 10);
    bench_bytetrack_update(&mut suite, 50);
    bench_bytetrack_update(&mut suite, 100);
    bench_bytetrack_100_frames(&mut suite);
    bench_kalman_predict(&mut suite);
    bench_kalman_update(&mut suite);

    suite.finish();
}
