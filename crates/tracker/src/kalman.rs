// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use nalgebra::{
    DVector, DefaultAllocator, Dyn, OMatrix, RealField, SVector, U1, U8, allocator::Allocator,
    convert, dimension::U4,
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
}
