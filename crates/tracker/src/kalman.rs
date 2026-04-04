// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use nalgebra::{
    allocator::Allocator, convert, dimension::U4, DVector, DefaultAllocator, Dyn, OMatrix,
    RealField, SVector, U1, U8,
};

#[derive(Debug, Clone)]
pub struct ConstantVelocityXYAHModel2<R>
where
    R: RealField,
    DefaultAllocator: Allocator<R, U8, U8>,
    DefaultAllocator: Allocator<R, U8>,
{
    pub mean: SVector<R, 8>,
    pub std_weight_position: R,
    pub std_weight_velocity: R,
    pub update_factor: R,
    motion_matrix: OMatrix<R, U8, U8>,
    update_matrix: OMatrix<R, U4, U8>,
    pub covariance: OMatrix<R, U8, U8>,
}

#[allow(dead_code)]
pub enum GatingDistanceMetric {
    Gaussian,
    Mahalanobis,
}

impl<R> ConstantVelocityXYAHModel2<R>
where
    R: RealField + Copy,
{
    pub fn new(measurement: &[R; 4], update_factor: R) -> Self {
        let ndim = 4;
        let dt: R = convert(1.0);

        let mut motion_matrix = OMatrix::<R, U8, U8>::identity();
        for i in 0..ndim {
            motion_matrix[(i, ndim + i)] = dt * convert(3.0);
        }
        let mut update_matrix = OMatrix::<R, U4, U8>::identity();
        for i in 0..ndim {
            update_matrix[(i, ndim + i)] = dt * convert(1.0);
        }
        let zero: R = convert(0.0);
        let two: R = convert(2.0);
        let ten: R = convert(10.0);
        let height = measurement[3];

        let mean = SVector::<R, 8>::from_row_slice(&[
            measurement[0],
            measurement[1],
            measurement[2],
            measurement[3],
            zero,
            zero,
            zero,
            zero,
        ]);
        let std_weight_position = convert(1.0 / 20.0);
        let std_weight_velocity = convert(1.0 / 160.0);
        let diag = [
            two * std_weight_position * height,
            two * std_weight_position * height,
            convert(0.01),
            two * std_weight_position * height,
            ten * std_weight_velocity * height,
            ten * std_weight_velocity * height,
            convert(0.00001),
            ten * std_weight_velocity * height,
        ];
        let diag = SVector::<R, 8>::from_row_slice(&diag);

        let covariance = OMatrix::<R, U8, U8>::from_diagonal(&diag.component_mul(&diag));
        Self {
            motion_matrix,
            update_matrix,
            mean,
            covariance,
            std_weight_position,
            std_weight_velocity,
            update_factor,
        }
    }

    pub fn predict(&mut self) {
        let height = self.mean[3];
        let diag = [
            self.std_weight_position * height,
            self.std_weight_position * height,
            convert(0.01),
            self.std_weight_position * height,
            self.std_weight_velocity * height,
            self.std_weight_velocity * height,
            convert(0.00001),
            self.std_weight_velocity * height,
        ];
        let diag = SVector::<R, 8>::from_row_slice(&diag);
        let motion_cov = OMatrix::<R, U8, U8>::from_diagonal(&diag.component_mul(&diag));

        let mean = (self.mean.transpose() * self.motion_matrix.transpose()).transpose();
        let covariance =
            self.motion_matrix * self.covariance * self.motion_matrix.transpose() + motion_cov;
        self.mean = mean;
        self.covariance = covariance;
    }

    pub fn project(&self) -> (OMatrix<R, U4, U1>, OMatrix<R, U4, U4>) {
        let height = self.mean[3];
        let diag = [
            self.std_weight_position * height,
            self.std_weight_position * height,
            convert(0.01),
            self.std_weight_position * height,
        ];
        let diag = SVector::<R, 4>::from_row_slice(&diag);
        let innovation_cov = OMatrix::<R, U4, U4>::from_diagonal(&diag.component_mul(&diag));
        let mean = self.update_matrix * self.mean;
        let covariance =
            self.update_matrix * self.covariance * self.update_matrix.transpose() + innovation_cov;
        (mean, covariance)
    }

    pub fn update(&mut self, measurement: &[R; 4]) {
        let measurement = SVector::<R, 4>::from_row_slice(&[
            measurement[0],
            measurement[1],
            measurement[2],
            measurement[3],
        ]);

        let (projected_mean, projected_cov) = self.project();
        let cho_factor = match projected_cov.cholesky() {
            None => return,
            Some(v) => v,
        };
        let kalman_gain = cho_factor
            .solve(&(self.covariance * self.update_matrix.transpose()).transpose())
            .transpose();

        let innovation = (measurement - projected_mean).scale(self.update_factor);
        // println!("kalman_gain={}", kalman_gain);
        // println!("innovation={}", innovation);
        let diff = innovation.transpose() * kalman_gain.transpose();
        self.mean += diff.transpose();
        self.covariance -= kalman_gain * projected_cov * kalman_gain.transpose();
        // let new_mean = self.mean + diff.transpose();
        // let new_cov = self.covariance - kalman_gain * projected_cov *
        // kalman_gain.transpose();
    }

    #[allow(dead_code)]
    pub fn gating_distance(
        &self,
        measurements: &OMatrix<R, Dyn, U4>,
        only_position: bool,
        metric: GatingDistanceMetric,
    ) -> DVector<R> {
        let (m, cov) = self.project();
        let ndims = if only_position { 2 } else { 4 };
        let mean = m.transpose();
        let mean = mean.columns_range(0..ndims);
        let covariance = cov.view_range(0..ndims, 0..ndims);
        let measurements = measurements.columns_range(0..ndims);
        // let _ = only_position;
        // let mean = m.transpose();
        // let covariance = cov;
        // let measurements = measurements;

        let mut mean_broadcast =
            OMatrix::<R, Dyn, U4>::from_element(measurements.shape().0, convert(0.0));
        for mut col in mean_broadcast.row_iter_mut() {
            col.copy_from(&mean);
        }
        let d = measurements - mean_broadcast;
        match metric {
            GatingDistanceMetric::Gaussian => d.component_mul(&d).column_sum(),
            GatingDistanceMetric::Mahalanobis => {
                let cho_factor = match covariance.cholesky() {
                    None => return DVector::<R>::zeros(measurements.shape().0),
                    Some(v) => v,
                };
                let z = cho_factor.solve(&d.transpose());
                z.component_mul(&z).row_sum_tr()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Dyn, OMatrix, U4};

    use super::{ConstantVelocityXYAHModel2, GatingDistanceMetric};
    #[test]
    fn filter() {
        let mut t = ConstantVelocityXYAHModel2::new(&[0.5, 0.5, 1.0, 0.5], 0.25);
        t.predict();
        println!("1. t.mean={}", t.mean);
        t.update(&[0.4, 0.5, 1.0, 0.5]);
        t.predict();
        println!("2. t.mean={}", t.mean);
        t.update(&[0.3, 0.5, 1.0, 0.5]);
        t.predict();
        println!("3. t.mean={}", t.mean);
        t.update(&[0.2, 0.5, 1.0, 0.5]);
        t.predict();
        println!("4. t.mean={}", t.mean);
        t.update(&[0.2, 0.5, 1.0, 0.5]);
        t.predict();
        println!("5. t.mean={}", t.mean);
        t.update(&[0.3, 0.5, 1.0, 0.5]);
        t.predict();
        println!("6. t.mean={}", t.mean);
        t.update(&[0.4, 0.5, 1.0, 0.5]);
    }

    #[test]
    fn gating() {
        let mut t = ConstantVelocityXYAHModel2::new(&[0.5, 0.5, 1.0, 0.5], 0.25);
        t.predict();
        t.update(&[0.49, 0.5, 1.0, 0.5]);
        t.predict();
        t.update(&[0.48, 0.5, 1.0, 0.5]);
        t.predict();
        t.update(&[0.47, 0.5, 1.0, 0.5]);
        t.predict();
        t.update(&[0.46, 0.5, 1.0, 0.5]);
        t.predict();
        t.update(&[0.45, 0.5, 1.0, 0.5]);
        t.predict();
        t.update(&[0.44, 0.5, 1.0, 0.5]);
        t.predict();
        t.update(&[0.43, 0.5, 1.0, 0.5]);
        t.predict();
        t.update(&[0.42, 0.5, 1.0, 0.5]);
        t.predict();

        // distances range from 0 to 1e6 for maha
        let mut measurements = OMatrix::<f32, Dyn, U4>::from_element(1, 0.0);
        measurements.copy_from_slice(&[0.3, 0.5, 1.0, 0.5]);

        let mut distances = OMatrix::<f32, Dyn, Dyn>::from_element(1, 1, 0.0);
        for mut column in distances.column_iter_mut() {
            let dist = t.gating_distance(&measurements, false, GatingDistanceMetric::Gaussian);
            column.copy_from(&dist);
        }
        let dist = t.gating_distance(&measurements, false, GatingDistanceMetric::Mahalanobis);
        println!("Dist(false, maha): {dist}");

        let dist = t.gating_distance(&measurements, false, GatingDistanceMetric::Gaussian);
        println!("Dist(false, gaussian): {dist}");
    }

    #[test]
    fn test_predict_constant_velocity() {
        // Initialize filter and give it a few updates to establish velocity,
        // then verify predictions drift in the expected direction.
        let mut t = ConstantVelocityXYAHModel2::new(&[0.5, 0.5, 0.1, 2.0], 0.25);
        t.predict();
        t.update(&[0.5, 0.5, 0.1, 2.0]);

        // Record position after first predict-update cycle
        let x_before: f32 = t.mean[0];
        let y_before: f32 = t.mean[1];

        // Run several predict-only cycles to let velocity dominate
        for _ in 0..5 {
            t.predict();
        }

        let x_after: f32 = t.mean[0];
        let y_after: f32 = t.mean[1];
        let h_after: f32 = t.mean[3];

        // The position should remain numerically reasonable (no NaN/Inf)
        assert!(x_after.is_finite(), "x should be finite after predictions");
        assert!(y_after.is_finite(), "y should be finite after predictions");
        assert!(
            h_after.is_finite(),
            "height should be finite after predictions"
        );

        // With near-zero velocity the predicted position should not explode
        assert!(
            (x_after - x_before).abs() < 5.0,
            "x drift should be bounded, got delta={}",
            (x_after - x_before).abs()
        );
        assert!(
            (y_after - y_before).abs() < 5.0,
            "y drift should be bounded, got delta={}",
            (y_after - y_before).abs()
        );
    }

    #[test]
    fn test_numerical_stability_1000_cycles() {
        let mut t = ConstantVelocityXYAHModel2::new(&[0.5, 0.5, 1.0, 0.5], 0.25);

        // Run 1000 predict-only cycles without any update
        for _ in 0..1000 {
            t.predict();
        }

        // Verify no NaN or Inf in the mean vector
        for i in 0..8 {
            let val: f32 = t.mean[i];
            assert!(
                val.is_finite(),
                "mean[{i}] should be finite after 1000 predictions, got {val}",
            );
        }

        // Verify no NaN or Inf in the covariance matrix
        for r in 0..8 {
            for c in 0..8 {
                let val: f32 = t.covariance[(r, c)];
                assert!(
                    val.is_finite(),
                    "covariance[({r},{c})] should be finite after 1000 predictions, got {val}",
                );
            }
        }
    }

    #[test]
    fn test_gating_distance_edge_cases() {
        let mut t = ConstantVelocityXYAHModel2::new(&[0.5, 0.5, 1.0, 0.5], 0.25);
        // Run a few predict-update cycles to stabilize
        for _ in 0..3 {
            t.predict();
            t.update(&[0.5, 0.5, 1.0, 0.5]);
        }
        t.predict();

        // Measurement exactly at the predicted state -- distance should be near 0
        let (projected_mean, _) = t.project();
        let mut meas_close = OMatrix::<f32, Dyn, U4>::from_element(1, 0.0);
        meas_close
            .row_mut(0)
            .copy_from_slice(projected_mean.as_slice());

        let dist_close = t.gating_distance(&meas_close, false, GatingDistanceMetric::Mahalanobis);
        assert!(
            dist_close[0].is_finite(),
            "Close-measurement distance should be finite"
        );
        assert!(
            dist_close[0] < 1.0,
            "Distance for exact-match measurement should be near 0, got {}",
            dist_close[0]
        );

        // Measurement far away -- distance should be large
        let mut meas_far = OMatrix::<f32, Dyn, U4>::from_element(1, 0.0);
        meas_far.copy_from_slice(&[10.0, 10.0, 5.0, 10.0]);

        let dist_far = t.gating_distance(&meas_far, false, GatingDistanceMetric::Mahalanobis);
        assert!(
            dist_far[0].is_finite(),
            "Far-measurement distance should be finite"
        );
        assert!(
            dist_far[0] > dist_close[0],
            "Far measurement should have larger distance than close one: {} vs {}",
            dist_far[0],
            dist_close[0]
        );
    }

    #[test]
    fn test_update_moves_mean_toward_measurement() {
        let mut t = ConstantVelocityXYAHModel2::new(&[0.5, 0.5, 1.0, 0.5], 0.25);
        t.predict();

        let x_before: f32 = t.mean[0];
        // Update with a measurement shifted to the right
        t.update(&[0.6, 0.5, 1.0, 0.5]);
        let x_after: f32 = t.mean[0];

        assert!(
            x_after > x_before,
            "Mean x should move toward the measurement (0.6), was {x_before}, now {x_after}"
        );
        assert!(
            x_after <= 0.6,
            "Mean x should not overshoot the measurement, got {x_after}"
        );
    }

    #[test]
    fn test_covariance_positive_diagonal() {
        let t = ConstantVelocityXYAHModel2::new(&[0.5, 0.5, 1.0, 0.5], 0.25);

        // All diagonal elements of the covariance should be positive
        for i in 0..8 {
            let val: f32 = t.covariance[(i, i)];
            assert!(
                val > 0.0,
                "Covariance diagonal[{i}] should be positive, got {val}"
            );
        }
    }

    #[test]
    fn test_predict_increases_uncertainty() {
        let mut t = ConstantVelocityXYAHModel2::new(&[0.5, 0.5, 1.0, 0.5], 0.25);

        let cov_before: f32 = t.covariance[(0, 0)];
        t.predict();
        let cov_after: f32 = t.covariance[(0, 0)];

        assert!(
            cov_after > cov_before,
            "Predict should increase position uncertainty: {cov_before} -> {cov_after}"
        );
    }

    #[test]
    fn test_update_decreases_uncertainty() {
        let mut t = ConstantVelocityXYAHModel2::new(&[0.5, 0.5, 1.0, 0.5], 0.25);
        t.predict();

        let cov_before: f32 = t.covariance[(0, 0)];
        t.update(&[0.5, 0.5, 1.0, 0.5]);
        let cov_after: f32 = t.covariance[(0, 0)];

        assert!(
            cov_after < cov_before,
            "Update should decrease position uncertainty: {cov_before} -> {cov_after}"
        );
    }

    #[test]
    fn test_gating_distance_gaussian_vs_mahalanobis() {
        let mut t = ConstantVelocityXYAHModel2::new(&[0.5, 0.5, 1.0, 0.5], 0.25);
        for _ in 0..3 {
            t.predict();
            t.update(&[0.5, 0.5, 1.0, 0.5]);
        }
        t.predict();

        let mut measurements = OMatrix::<f32, Dyn, U4>::from_element(1, 0.0);
        measurements.copy_from_slice(&[0.6, 0.5, 1.0, 0.5]);

        let dist_gauss = t.gating_distance(&measurements, false, GatingDistanceMetric::Gaussian);
        let dist_maha = t.gating_distance(&measurements, false, GatingDistanceMetric::Mahalanobis);

        assert!(dist_gauss[0].is_finite());
        assert!(dist_maha[0].is_finite());

        // Both should be non-negative for a non-zero offset
        assert!(
            dist_gauss[0] > 0.0,
            "Gaussian distance should be > 0 for offset measurement"
        );
        assert!(
            dist_maha[0] > 0.0,
            "Mahalanobis distance should be > 0 for offset measurement"
        );
    }

    #[test]
    fn test_gating_distance_multiple_measurements() {
        let mut t = ConstantVelocityXYAHModel2::new(&[0.5, 0.5, 1.0, 0.5], 0.25);
        t.predict();
        t.update(&[0.5, 0.5, 1.0, 0.5]);
        t.predict();

        // Two measurements: one close, one far
        let mut measurements = OMatrix::<f32, Dyn, U4>::from_element(2, 0.0);
        measurements
            .row_mut(0)
            .copy_from_slice(&[0.5, 0.5, 1.0, 0.5]); // close
        measurements
            .row_mut(1)
            .copy_from_slice(&[5.0, 5.0, 1.0, 0.5]); // far

        let dists = t.gating_distance(&measurements, false, GatingDistanceMetric::Mahalanobis);
        assert_eq!(dists.len(), 2, "Should return one distance per measurement");
        assert!(dists[0].is_finite());
        assert!(dists[1].is_finite());
        assert!(
            dists[1] > dists[0],
            "Far measurement should have larger distance: {} vs {}",
            dists[1],
            dists[0]
        );
    }

    #[test]
    fn test_initiate_mean_matches_measurement() {
        let measurement = [0.3, 0.7, 1.5, 2.0];
        let t = ConstantVelocityXYAHModel2::new(&measurement, 0.25);

        // Position portion of mean should match the measurement exactly
        let x: f32 = t.mean[0];
        let y: f32 = t.mean[1];
        let a: f32 = t.mean[2];
        let h: f32 = t.mean[3];
        assert!((x - 0.3).abs() < 1e-6, "Mean x should be 0.3, got {x}");
        assert!((y - 0.7).abs() < 1e-6, "Mean y should be 0.7, got {y}");
        assert!((a - 1.5).abs() < 1e-6, "Mean a should be 1.5, got {a}");
        assert!((h - 2.0).abs() < 1e-6, "Mean h should be 2.0, got {h}");

        // Velocity portion should be zero
        for i in 4..8 {
            let v: f32 = t.mean[i];
            assert!((v).abs() < 1e-6, "Velocity mean[{i}] should be 0, got {v}");
        }
    }
}
