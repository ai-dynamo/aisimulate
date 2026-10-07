// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dimension-specialized stores keep the default two-axis arithmetic unchanged.
use super::super::estimator::RegressionFeatureAxis;
use super::*;

macro_rules! stores {
    ($($variant:ident: $d:literal, $s:literal);+ $(;)?) => {
        #[derive(Clone, Debug)]
        enum Fits { $($variant(BucketedRegression<$d, $s>)),+ }
        impl Fits {
            fn new(options: &ForwardPassPerfOptions, fit: &RegressionFitConfig, sampling: &RegressionSamplingConfig, dimensions: usize) -> Self {
                match dimensions { $($d => Self::$variant(BucketedRegression::configured(options, fit, sampling)),)+ _ => unreachable!("validated feature dimension") }
            }
            fn add(&mut self, x: [f64; 6], y: f64, bucket: Vec<f64>) -> bool {
                match self { $(Self::$variant(store) => store.add_projected(x[..$d].try_into().unwrap(), y, bucket)),+ }
            }
            fn predict(&self, x: [f64; 6]) -> Option<f64> {
                match self { $(Self::$variant(store) => store.predict(x[..$d].try_into().unwrap())),+ }
            }
            #[cfg(test)]
            fn mutations_since_rebuild(&self) -> usize { match self { $(Self::$variant(store) => store.mutations_since_rebuild()),+ } }
        }
        impl StoreStats for Fits {
            fn observation_count(&self) -> usize { match self { $(Self::$variant(store) => store.observation_count()),+ } }
            fn is_ready(&self) -> bool { match self { $(Self::$variant(store) => store.is_ready()),+ } }
        }
    };
}
stores! { One:1,2; Two:2,3; Three:3,4; Four:4,5; Five:5,6; Six:6,7; }

#[derive(Clone, Debug)]
pub(crate) struct SelectedRegression {
    fit_axes: Vec<RegressionFeatureAxis>,
    projection_axes: Vec<RegressionFeatureAxis>,
    sampling_indices: Vec<usize>,
    fits: Fits,
}
impl SelectedRegression {
    pub(super) fn new(
        options: &ForwardPassPerfOptions,
        fit: &RegressionFitConfig,
        sampling: &RegressionSamplingConfig,
    ) -> Self {
        let fit_axes = fit.linear.clone().unwrap_or_default().feature_axes;
        let mut projection_axes = fit_axes.clone();
        let sampling_indices = sampling
            .axes
            .iter()
            .map(|axis| {
                if let Some(index) = projection_axes.iter().position(|selected| selected == axis) {
                    index
                } else {
                    projection_axes.push(*axis);
                    projection_axes.len() - 1
                }
            })
            .collect();
        Self {
            fits: Fits::new(options, fit, sampling, fit_axes.len()),
            fit_axes,
            projection_axes,
            sampling_indices,
        }
    }
    pub(super) fn add(
        &mut self,
        scores: [f64; 2],
        target: f64,
        metrics: &[ForwardPassMetrics],
    ) -> Result<bool, AicError> {
        // Validate both projections before admitting data or changing statistics.
        let projected = feature_axes::project(scores, metrics, &self.projection_axes)?;
        let mut x = [0.0; 6];
        x[..self.fit_axes.len()].copy_from_slice(&projected[..self.fit_axes.len()]);
        let bucket = self
            .sampling_indices
            .iter()
            .map(|index| projected[*index].ln_1p())
            .collect();
        Ok(self.fits.add(x, target, bucket))
    }
    pub(super) fn predict(
        &self,
        scores: [f64; 2],
        metrics: &[ForwardPassMetrics],
    ) -> Result<Option<f64>, AicError> {
        let projected = feature_axes::project(scores, metrics, &self.fit_axes)?;
        Ok(self.fits.predict(projected[..6].try_into().unwrap()))
    }
    #[cfg(test)]
    pub(super) fn mutations_since_rebuild(&self) -> usize {
        self.fits.mutations_since_rebuild()
    }
}
impl StoreStats for SelectedRegression {
    fn observation_count(&self) -> usize {
        self.fits.observation_count()
    }
    fn is_ready(&self) -> bool {
        self.fits.is_ready()
    }
}
