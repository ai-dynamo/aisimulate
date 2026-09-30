// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::estimator::{
    LinearFitConfig, RegressionFeatureAxis as Axis, RegressionUpdatePolicy,
};
use super::*;

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-7 * expected.abs().max(1.0),
        "expected {expected}, got {actual}"
    );
}

fn snapshot<const D: usize>(fit: &LinearFit<D>) -> Vec<u64> {
    std::iter::once(fit.intercept.to_bits())
        .chain(fit.coefficients.map(f64::to_bits))
        .chain(fit.standardization.means.map(f64::to_bits))
        .chain(fit.standardization.scales.map(f64::to_bits))
        .chain(fit.standardization.active.map(u64::from))
        .collect()
}

fn retained<const D: usize, const S: usize>(
    store: &BucketedRegression<D, S>,
) -> Vec<RegressionObservation<D>> {
    store
        .samples
        .observations()
        .into_iter()
        .map(|(_, row)| row)
        .collect()
}

fn lazy_store(interval: Option<usize>) -> BucketedRegression {
    let options = ForwardPassPerfOptions::default();
    let fit = RegressionFitConfig {
        rebuild_interval: interval,
        linear: Some(LinearFitConfig {
            update_policy: RegressionUpdatePolicy::ErrorThreshold {
                relative_tolerance: 0.0,
                absolute_tolerance_ms: 1.0,
                window: 1,
                trigger: 1,
                cooldown: 1,
                startup_observations: 10,
            },
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut store =
        BucketedRegression::configured(&options, &fit, &RegressionSamplingConfig::default());
    for i in 0..10 {
        let x = [(i % 7 + 1) as f64, ((i * 3) % 11 + 1) as f64];
        assert!(store.add_observation(x, 5.0 + 2.0 * x[0] + 3.0 * x[1]));
    }
    assert!(store.is_ready());
    store
}

#[test]
fn lazy_publication_freezes_normalization_and_yields_to_error_or_full_rebuild() {
    let mut lazy = lazy_store(None);
    let published = snapshot(lazy.fit.as_ref().unwrap());
    let query = [20.0, 21.0];
    let prediction = lazy.predict(&query).unwrap();
    assert!(lazy.add_observation(query, prediction + 0.05));
    assert_eq!(snapshot(lazy.fit.as_ref().unwrap()), published);
    assert_eq!(
        lazy.predict(&query).unwrap().to_bits(),
        prediction.to_bits()
    );
    assert_eq!(lazy.observation_count(), 11);

    assert!(lazy.add_observation(query, prediction + 20.0));
    assert_ne!(snapshot(lazy.fit.as_ref().unwrap()), published);
    let batch = fit_regression_with_ridge(&retained(&lazy), lazy.min_observations, 1e-9).unwrap();
    close(
        lazy.predict(&query).unwrap(),
        batch.predict(&query).unwrap(),
    );

    let mut periodic = lazy_store(Some(12));
    let published = snapshot(periodic.fit.as_ref().unwrap());
    let mut rebuilt = false;
    for i in 0..12 {
        let x = [(i % 7 + 1) as f64, ((i * 5) % 11 + 1) as f64];
        let prior = periodic.predict(&x).unwrap();
        let due = periodic.mutations_since_rebuild() + 1 >= 12;
        // Below the error threshold; only the full-rebuild clock can publish.
        assert!(periodic.add_observation(x, prior + 0.05));
        if due {
            assert_eq!(periodic.mutations_since_rebuild(), 0);
            assert_ne!(snapshot(periodic.fit.as_ref().unwrap()), published);
            let batch =
                fit_regression_with_ridge(&retained(&periodic), periodic.min_observations, 1e-9)
                    .unwrap();
            close(
                periodic.predict(&query).unwrap(),
                batch.predict(&query).unwrap(),
            );
            rebuilt = true;
            break;
        }
        assert_eq!(snapshot(periodic.fit.as_ref().unwrap()), published);
    }
    assert!(rebuilt, "periodic rebuilding must bypass the lazy gate");
}

fn signed_grid<const D: usize, const S: usize>(sampling: RegressionSamplingConfig) {
    let options = ForwardPassPerfOptions::default();
    let axes = [Axis::Attention, Axis::Moe, Axis::Count, Axis::Past];
    let fit = RegressionFitConfig {
        linear: Some(LinearFitConfig {
            feature_axes: axes[..D].to_vec(),
            non_negative: false,
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut store = BucketedRegression::<D, S>::configured(&options, &fit, &sampling);
    let slopes = [1.5, -2.0, 0.25, 3.0];
    for i in 0..160 {
        let x: [f64; D] =
            std::array::from_fn(|a| ((i * [5, 7, 11, 13][a]) % [17, 19, 23, 29][a] + 1) as f64);
        let y = 120.0 + (0..D).map(|a| slopes[a] * x[a]).sum::<f64>();
        let bucket = sampling
            .axes
            .iter()
            .map(|axis| {
                let index = axes.iter().position(|candidate| candidate == axis).unwrap();
                x[index].ln_1p()
            })
            .collect();
        assert!(store.add_projected(x, y, bucket));
        assert!(
            store
                .samples
                .buckets
                .keys()
                .all(|key| key.len() == sampling.axes.len())
        );
        assert_eq!(
            store.observation_count(),
            (i + 1).min(sampling.max_observations)
        );
        if i >= 12 {
            let batch = fit_regression_with_constraints(
                &retained(&store),
                options.min_observations,
                options.regression_ridge_scale,
                false,
            )
            .unwrap();
            close(store.predict(&x).unwrap(), batch.predict(&x).unwrap());
        }
    }
    assert!(store.fit.as_ref().unwrap().coefficients[1] < 0.0);
    let query: [f64; D] = std::array::from_fn(|a| [8.0, 5.0, 12.0, 7.0][a]);
    // Hand oracle: 120 + 12 - 10 + 3 = 125; fourth-axis contribution is 21.
    close(
        store.predict(&query).unwrap(),
        if D == 3 { 125.0 } else { 146.0 },
    );
}

#[test]
fn fit_dimensions_and_signed_coefficients_are_independent_of_retention_grid() {
    signed_grid::<3, 4>(RegressionSamplingConfig {
        axes: vec![Axis::Count],
        bins_per_axis: vec![8],
        max_observations: 24,
    });
    signed_grid::<4, 5>(RegressionSamplingConfig {
        axes: vec![Axis::Count, Axis::Past, Axis::Attention],
        bins_per_axis: vec![2, 3, 2],
        max_observations: 24,
    });

    let sampling = RegressionSamplingConfig {
        axes: vec![Axis::Count],
        bins_per_axis: vec![4],
        max_observations: 24,
    };
    let fit = RegressionFitConfig {
        linear: Some(LinearFitConfig {
            feature_axes: vec![Axis::Attention, Axis::Moe, Axis::Count],
            non_negative: false,
            ..Default::default()
        }),
        ..Default::default()
    };
    let options = ForwardPassPerfOptions::default();
    let mut dependent = BucketedRegression::<3, 4>::configured(&options, &fit, &sampling);
    for i in 0..80 {
        let (a, b) = ((i % 17 + 1) as f64, ((i * 7) % 19 + 1) as f64);
        // No pair is perfectly correlated, but the third axis is their sum.
        let x = [a, b, a + b];
        assert!(dependent.add_projected(x, 120.0 + 2.0 * a - 3.0 * b, vec![x[2].ln_1p()]));
        if i >= 12 {
            let batch = fit_regression_with_constraints(
                &retained(&dependent),
                options.min_observations,
                options.regression_ridge_scale,
                false,
            )
            .unwrap();
            // Off-span predictions also need the same reference solution.
            for query in [x, [9.0, 4.0, 30.0], [3.0, 11.0, 2.0]] {
                close(
                    dependent.predict(&query).unwrap(),
                    batch.predict(&query).unwrap(),
                );
            }
            assert_eq!(dependent.mutations_since_rebuild(), 0);
        }
    }

    let options = ForwardPassPerfOptions::default();
    let mut legacy = BucketedRegression::new(&options, None);
    let mut configured: BucketedRegression = BucketedRegression::configured(
        &options,
        &RegressionFitConfig::default(),
        &RegressionSamplingConfig::default(),
    );
    for i in 0..60 {
        let x = [(i % 17 + 1) as f64, ((i * 7) % 23 + 1) as f64];
        let y = 5.0 + 2.0 * x[0] + 3.0 * x[1];
        assert_eq!(
            legacy.add_observation(x, y),
            configured.add_observation(x, y)
        );
        assert_eq!(legacy.observation_count(), configured.observation_count());
        assert_eq!(legacy.is_ready(), configured.is_ready());
        if legacy.is_ready() {
            close(legacy.predict(&x).unwrap(), configured.predict(&x).unwrap());
        }
    }
    let sorted = |store: &BucketedRegression| {
        let mut rows: Vec<_> = retained(store)
            .into_iter()
            .map(|r| (r.raw_x.map(f64::to_bits), r.observed_ms.to_bits()))
            .collect();
        rows.sort_unstable();
        rows
    };
    assert_eq!(sorted(&legacy), sorted(&configured));
}

fn metric(value: serde_json::Value) -> ForwardPassMetrics {
    serde_json::from_value(serde_json::json!({"scheduled_requests": value})).unwrap()
}

#[test]
fn request_features_are_explicit_and_projection_failures_do_not_mutate_store() {
    let scalar = [metric(serde_json::json!({"num_decode_requests": 3}))];
    let projected = feature_axes::project(
        [7.0, 11.0],
        &scalar,
        &[
            Axis::Attention,
            Axis::Moe,
            Axis::Count,
            Axis::LogCount,
            Axis::CountSquared,
        ],
    )
    .unwrap();
    assert_eq!(&projected[..3], &[7.0, 11.0, 3.0]);
    close(projected[3], 4.0_f64.ln());
    assert_eq!(projected[4], 9.0);

    let requests = [
        metric(
            serde_json::json!({"num_decode_requests": 2, "sum_decode_kv_tokens": 10, "extend_lengths": [1,1], "past_kv_lengths": [3,7]}),
        ),
        metric(
            serde_json::json!({"num_prefill_requests": 1, "sum_prefill_tokens": 2, "sum_prefill_kv_tokens": 4, "extend_lengths": [2], "past_kv_lengths": [4]}),
        ),
        ForwardPassMetrics::default(),
    ];
    let projected = feature_axes::project(
        [0.0, 0.0],
        &requests,
        &[
            Axis::Count,
            Axis::Extend,
            Axis::Past,
            Axis::AttentionPairs,
            Axis::PastCvSquared,
            Axis::MeanPast,
        ],
    )
    .unwrap();
    // Requests (extend,past)=(1,3),(1,7),(2,4): F=3.5+7.5+10;
    // CV_past^2=(3*(9+49+16)/(3+7+4)^2)-1=13/98.
    for (actual, expected) in
        projected
            .into_iter()
            .zip([3.0, 4.0, 14.0, 21.0, 13.0 / 98.0, 14.0 / 3.0])
    {
        close(actual, expected);
    }

    let sampling = RegressionSamplingConfig {
        axes: vec![Axis::Past],
        bins_per_axis: vec![4],
        max_observations: 64,
    };
    let mut store = RegressionStore::new(
        &ForwardPassPerfOptions::default(),
        &RegressionFitConfig::default(),
        &sampling,
    );
    assert!(matches!(&store, RegressionStore::Selected(_)));
    for i in 0..10 {
        let x = [(i % 7 + 1) as f64, ((i * 3) % 11 + 1) as f64];
        assert!(
            store
                .add_metrics(x, 5.0 + 2.0 * x[0] + 3.0 * x[1], &requests)
                .unwrap()
        );
    }
    assert!(store.is_ready());
    // Sampling alone requires request lists; prediction only projects fit axes.
    assert!(
        store
            .predict_metrics([3.0, 5.0], &scalar)
            .unwrap()
            .is_some()
    );
    let before = format!("{store:?}");
    assert!(store.add_metrics([3.0, 5.0], 25.0, &scalar).is_err());
    let malformed = [metric(
        serde_json::json!({"num_decode_requests": 2, "extend_lengths": [1,1], "past_kv_lengths": [3]}),
    )];
    assert!(store.add_metrics([3.0, 5.0], 25.0, &malformed).is_err());
    assert!(!store.add_metrics([3.0, 5.0], f64::NAN, &requests).unwrap());
    assert_eq!(format!("{store:?}"), before);
}
