// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Expected values are hand-derived from scheduled counters or are the known
//! linear functions that generated the targets, never refits of the same rows.
use super::super::super::estimator::{LinearFitConfig, RegressionUpdatePolicy};
use super::super::super::learned::{
    AGGREGATE_FEATURE_NAMES, FPM5_AGGREGATE_INDICES, IterationFeatureVector,
};
use super::*;
use crate::ScheduledRequestMetrics;

/// Python `aisimulate_core.sdk.fpm_learned.FEATURE_PRESETS["fpm5"]`.
const FPM5_PRESET: [&str; D] = [
    "num_active_ranks",
    "num_prefill_requests",
    "sum_prefill_tokens",
    "sum_prefill_kv_tokens",
    "num_decode_requests",
    "sum_decode_kv_tokens",
    "max_rank_prefill_tokens",
    "max_rank_decode_kv_tokens",
    "max_rank_num_decode_requests",
    "mean_prefill_chunk",
    "mean_prefill_kv",
    "prefill_attention_pairs",
    "mean_decode_kv",
    "log1p_sum_prefill_tokens",
    "log1p_sum_prefill_kv_tokens",
    "log1p_sum_decode_kv_tokens",
    "log1p_prefill_attention_pairs",
];

/// Positive raw weights giving each feature a comparable contribution.
const WEIGHTS: [f64; D] = [
    0.5, 0.3, 2e-3, 1e-4, 0.02, 5e-5, 1e-3, 1e-4, 0.01, 1e-3, 1e-4, 1e-8, 1e-4, 0.1, 0.05, 0.2,
    0.03,
];

struct Lcg(u64);

impl Lcg {
    fn between(&mut self, low: u32, high: u32) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        low + ((self.0 >> 33) % u64::from(high - low)) as u32
    }
}

fn rank(np: u32, ptok: u32, pkv: u32, nd: u32, dkv: u32) -> ForwardPassMetrics {
    ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_prefill_requests: np,
            sum_prefill_tokens: ptok,
            sum_prefill_kv_tokens: pkv,
            num_decode_requests: nd,
            sum_decode_kv_tokens: dkv,
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Four attention-DP ranks; about a third are idle, so the active rank count
/// varies, but at least one rank always has work.
fn dp4_iteration(lcg: &mut Lcg) -> Vec<ForwardPassMetrics> {
    loop {
        let ranks = (0..4)
            .map(|_| {
                if lcg.between(0, 3) == 0 {
                    return rank(0, 0, 0, 0, 0);
                }
                let np = lcg.between(0, 4);
                let (ptok, pkv) = if np > 0 {
                    (np * lcg.between(16, 2_048), lcg.between(0, 8_192))
                } else {
                    (0, 0)
                };
                let nd = lcg.between(0, 64);
                rank(np, ptok, pkv, nd, nd * lcg.between(32, 4_096))
            })
            .collect::<Vec<_>>();
        if fpm5_features(&ranks)[0] > 0.0 {
            return ranks;
        }
    }
}

fn truth(features: &[f64; D], weights: &[f64; D]) -> f64 {
    3.0 + features
        .iter()
        .zip(weights)
        .map(|(value, weight)| value * weight)
        .sum::<f64>()
}

fn store(
    min_observations: usize,
    max_observations: usize,
    linear: LinearFitConfig,
) -> AggregateRegression {
    let options = ForwardPassPerfOptions {
        min_observations,
        ..ForwardPassPerfOptions::default()
    };
    let fit = RegressionFitConfig {
        linear: Some(LinearFitConfig {
            feature_set: Some(RegressionFeatureSet::Fpm5),
            ..linear
        }),
        ..RegressionFitConfig::default()
    };
    let sampling = RegressionSamplingConfig {
        max_observations,
        ..RegressionSamplingConfig::default()
    };
    AggregateRegression::new(&options, &fit, &sampling)
}

fn add(store: &mut AggregateRegression, metrics: &[ForwardPassMetrics], target: f64) {
    // Retention scores only bucket the row; the fit never reads them.
    let features = fpm5_features(metrics);
    assert!(
        store
            .add([features[5], features[2]], target, metrics)
            .unwrap()
    );
}

fn close(actual: f64, expected: f64, relative: f64) {
    assert!(
        (actual - expected).abs() <= relative * expected.abs().max(1.0),
        "expected {expected}, got {actual}"
    );
}

#[test]
fn fpm5_features_are_the_learned_aggregate_block_in_preset_order() {
    assert_eq!(
        FPM5_AGGREGATE_INDICES.map(|index| AGGREGATE_FEATURE_NAMES[index]),
        FPM5_PRESET
    );
    // Rank 0: two prefill chunks (300 new, 100 cached tokens) and four decodes
    // reading 4000 KV tokens. Rank 1: six decodes reading 1200. Rank 2 is idle.
    let metrics = [
        rank(2, 300, 100, 4, 4_000),
        rank(0, 0, 0, 6, 1_200),
        rank(0, 0, 0, 0, 0),
    ];
    // Balanced partition e = 300 / 2, p = 100 / 2: 2 * 150 * (50 + 151 / 2).
    let pairs = 37_650.0;
    let expected = [
        2.0,
        2.0,
        300.0,
        100.0,
        10.0,
        5_200.0,
        300.0,
        4_000.0,
        6.0,
        150.0,
        50.0,
        pairs,
        520.0,
        300.0_f64.ln_1p(),
        100.0_f64.ln_1p(),
        5_200.0_f64.ln_1p(),
        f64::ln_1p(pairs),
    ];
    let features = fpm5_features(&metrics);
    let learned = IterationFeatureVector::from_metrics(&metrics);
    for (index, name) in FPM5_PRESET.iter().enumerate() {
        assert_eq!(
            features[index].to_bits(),
            expected[index].to_bits(),
            "{name}"
        );
        assert_eq!(
            learned.get(name).unwrap().to_bits(),
            features[index].to_bits(),
            "{name}"
        );
    }
    let mut lcg = Lcg(7);
    for _ in 0..50 {
        let metrics = dp4_iteration(&mut lcg);
        let learned = IterationFeatureVector::from_metrics(&metrics);
        let features = fpm5_features(&metrics);
        for (feature, index) in features.iter().zip(FPM5_AGGREGATE_INDICES) {
            assert_eq!(feature.to_bits(), learned.values[index].to_bits());
        }
    }
}

#[test]
fn recovers_a_linear_function_of_the_multi_rank_aggregates() {
    let mut lcg = Lcg(11);
    let mut store = store(5, 256, LinearFitConfig::default());
    for row in 1..=120 {
        let metrics = dp4_iteration(&mut lcg);
        add(
            &mut store,
            &metrics,
            truth(&fpm5_features(&metrics), &WEIGHTS),
        );
        // Seventeen varying features need eighteen rows.
        assert_eq!(store.is_ready(), row > D, "row {row}");
    }
    let fit = store.fit.as_ref().unwrap();
    assert!(fit.standardization.active.iter().all(|active| *active));
    assert!(
        fit.coefficients
            .iter()
            .all(|coefficient| *coefficient > 0.0)
    );
    for _ in 0..40 {
        let query = dp4_iteration(&mut lcg);
        close(
            store.predict(&query).unwrap(),
            truth(&fpm5_features(&query), &WEIGHTS),
            1e-6,
        );
    }
    for non_negative in [false, true] {
        let mut lcg = Lcg(11);
        let mut signed = self::store(
            5,
            256,
            LinearFitConfig {
                non_negative,
                ..LinearFitConfig::default()
            },
        );
        for _ in 0..120 {
            let metrics = dp4_iteration(&mut lcg);
            add(
                &mut signed,
                &metrics,
                truth(&fpm5_features(&metrics), &WEIGHTS),
            );
        }
        let query = dp4_iteration(&mut lcg);
        close(
            signed.predict(&query).unwrap(),
            truth(&fpm5_features(&query), &WEIGHTS),
            1e-6,
        );
    }
}

#[test]
fn single_rank_duplicates_and_constant_ranks_are_inactive() {
    // One rank: `max_rank_*` equal their sums and the rank count is constant.
    // Weight on a duplicate is indistinguishable from weight on its sum.
    let mut lcg = Lcg(23);
    let mut iteration = || {
        let np = lcg.between(1, 4);
        let nd = lcg.between(1, 128);
        vec![rank(
            np,
            np * lcg.between(16, 2_048),
            lcg.between(0, 8_192),
            nd,
            nd * lcg.between(32, 4_096),
        )]
    };
    let mut store = store(5, 64, LinearFitConfig::default());
    for _ in 0..64 {
        let metrics = iteration();
        add(
            &mut store,
            &metrics,
            truth(&fpm5_features(&metrics), &WEIGHTS),
        );
    }
    let fit = store.fit.as_ref().unwrap();
    let inactive = [0, 6, 7, 8];
    for (index, name) in FPM5_PRESET.iter().enumerate() {
        assert_eq!(
            fit.standardization.active[index],
            !inactive.contains(&index),
            "{name}"
        );
        if inactive.contains(&index) {
            assert_eq!(fit.coefficients[index], 0.0, "{name}");
        }
    }
    for _ in 0..20 {
        let query = iteration();
        close(
            store.predict(&query).unwrap(),
            truth(&fpm5_features(&query), &WEIGHTS),
            1e-6,
        );
    }
}

#[test]
fn readiness_needs_min_observations_and_one_row_per_active_feature() {
    // Single-rank decode varies four features: requests, KV tokens, their mean
    // and log1p(KV). Rank count, prefill and duplicate maxima stay inactive.
    let rows = [
        (1, 100),
        (2, 500),
        (3, 600),
        (4, 2_000),
        (5, 3_500),
        (6, 3_000),
        (7, 9_000),
        (8, 4_000),
        (9, 12_000),
    ];
    let target = |nd: u32, dkv: u32| 1.0 + 0.1 * f64::from(nd) + 1e-3 * f64::from(dkv);
    for (min_observations, ready_from) in [(2, 5), (5, 5), (8, 8)] {
        let mut store = store(min_observations, 64, LinearFitConfig::default());
        for (row, (nd, dkv)) in rows.iter().copied().enumerate() {
            add(&mut store, &[rank(0, 0, 0, nd, dkv)], target(nd, dkv));
            assert_eq!(
                store.is_ready(),
                row + 1 >= ready_from,
                "min {min_observations}, row {}",
                row + 1
            );
        }
        let active = store.fit.as_ref().unwrap().standardization.active;
        assert_eq!(active.iter().filter(|active| **active).count(), 4);
    }
}

#[test]
fn a_ready_store_keeps_serving_when_new_traffic_activates_more_aggregates() {
    let target = |nd: u32, dkv: u32| 1.0 + 0.1 * f64::from(nd) + 1e-3 * f64::from(dkv);
    let mut store = store(5, 64, LinearFitConfig::default());
    for (nd, dkv) in [(1, 100), (2, 500), (3, 600), (4, 2_000), (5, 3_500)] {
        add(&mut store, &[rank(0, 0, 0, nd, dkv)], target(nd, dkv));
    }
    assert!(store.is_ready());
    let query = [rank(0, 0, 0, 4, 1_000)];
    let served = store.predict(&query).unwrap();
    // A two-rank iteration activates the rank count and the max-rank
    // aggregates: seven active features with six retained rows.
    let ranks = [rank(0, 0, 0, 3, 900), rank(0, 0, 0, 2, 700)];
    add(&mut store, &ranks, target(5, 1_600));
    assert!(store.is_ready());
    assert_eq!(store.predict(&query), Some(served));
}

#[test]
fn predictions_are_finite_and_floored_for_signed_and_nonnegative_fits() {
    // Latency falls with batch size, so a signed fit extrapolates below zero.
    let target = |nd: u32, dkv: u32| 50.0 - 0.5 * f64::from(nd) + 1e-4 * f64::from(dkv);
    let far = [rank(0, 0, 0, 4_000, 4_000)];
    for non_negative in [false, true] {
        let mut store = store(
            5,
            64,
            LinearFitConfig {
                non_negative,
                ..LinearFitConfig::default()
            },
        );
        for nd in 1..=40 {
            let dkv = nd * (100 + 37 * (nd % 5));
            add(&mut store, &[rank(0, 0, 0, nd, dkv)], target(nd, dkv));
        }
        let fit = store.fit.as_ref().unwrap();
        assert!(fit.coefficients.iter().all(|c| c.is_finite()));
        assert_eq!(
            fit.coefficients.iter().all(|c| *c >= 0.0),
            non_negative,
            "only signed fits may use negative weights"
        );
        let prediction = store.predict(&far).unwrap();
        assert!(prediction.is_finite() && prediction >= MIN_POSITIVE_PREDICTION_MS);
        if !non_negative {
            assert_eq!(prediction, MIN_POSITIVE_PREDICTION_MS);
        }
    }
}

#[test]
fn nonnegative_least_squares_matches_hand_solved_kkt_points() {
    // Unconstrained solution (1, -1); the constrained optimum drops slope 2.
    let solution = nonnegative_least_squares(&[vec![1.0, 0.0], vec![0.0, 1.0]], &[1.0, -1.0]);
    assert_eq!(solution, Some(vec![1.0, 0.0]));
    // Correlated: [[2, 1], [1, 2]] x = [1, -1] gives (1, -1). With x2 = 0,
    // x1 = 1/2 and the x2 gradient -1 - 1/2 is negative, so (1/2, 0) is KKT.
    let solution =
        nonnegative_least_squares(&[vec![2.0, 1.0], vec![1.0, 2.0]], &[1.0, -1.0]).unwrap();
    close(solution[0], 0.5, 1e-15);
    assert_eq!(solution[1], 0.0);
    // Interior optimum: (1, 2) solves [[2, 1], [1, 2]] x = [4, 5].
    let solution =
        nonnegative_least_squares(&[vec![2.0, 1.0], vec![1.0, 2.0]], &[4.0, 5.0]).unwrap();
    close(solution[0], 1.0, 1e-14);
    close(solution[1], 2.0, 1e-14);
    // Every gradient nonpositive at zero: the origin is optimal.
    assert_eq!(
        nonnegative_least_squares(&[vec![1.0]], &[-3.0]),
        Some(vec![0.0])
    );
}

#[test]
fn lazy_updates_defer_publication_until_errors_exceed_the_threshold() {
    let mut store = store(
        5,
        64,
        LinearFitConfig {
            update_policy: RegressionUpdatePolicy::ErrorThreshold {
                relative_tolerance: 0.0,
                absolute_tolerance_ms: 1.0,
                window: 1,
                trigger: 1,
                cooldown: 1,
                startup_observations: 8,
            },
            ..LinearFitConfig::default()
        },
    );
    let target = |nd: u32, dkv: u32| 2.0 + 0.1 * f64::from(nd) + 1e-3 * f64::from(dkv);
    for nd in 1..=8 {
        let dkv = nd * (300 + 41 * (nd % 3));
        add(&mut store, &[rank(0, 0, 0, nd, dkv)], target(nd, dkv));
    }
    let published = format!("{:?}", store.fit);
    let query = [rank(0, 0, 0, 20, 9_000)];
    let prediction = store.predict(&query).unwrap();
    add(&mut store, &query, prediction + 0.5);
    assert_eq!(format!("{:?}", store.fit), published);
    assert_eq!(store.observation_count(), 9);
    add(&mut store, &query, prediction + 20.0);
    assert_ne!(format!("{:?}", store.fit), published);
}

#[test]
fn routing_retention_and_invalid_inputs_leave_other_fits_unchanged() {
    let options = ForwardPassPerfOptions::default();
    let sampling = RegressionSamplingConfig::default();
    let default = RegressionStore::new(&options, &RegressionFitConfig::default(), &sampling);
    assert!(matches!(default, RegressionStore::Linear(_)));
    let fit = RegressionFitConfig {
        linear: Some(LinearFitConfig {
            feature_set: Some(RegressionFeatureSet::Fpm5),
            ..LinearFitConfig::default()
        }),
        ..RegressionFitConfig::default()
    };
    let aggregate = RegressionStore::new(&options, &fit, &sampling);
    assert!(matches!(aggregate, RegressionStore::Aggregate(_)));
    assert!(aggregate.spline_diagnostics().is_none());

    // Request-list retention rejects scalar-only telemetry before mutation.
    let request_axes = RegressionSamplingConfig {
        axes: vec![RegressionFeatureAxis::Past],
        bins_per_axis: vec![4],
        max_observations: 64,
    };
    let mut store = RegressionStore::new(&options, &fit, &request_axes);
    let scalar = [rank(0, 0, 0, 2, 600)];
    let before = format!("{store:?}");
    assert!(store.add_metrics([1.0, 2.0], 5.0, &scalar).is_err());
    let mut with_lists = rank(0, 0, 0, 2, 600);
    with_lists.scheduled_requests.extend_lengths = Some(vec![1, 1]);
    with_lists.scheduled_requests.past_kv_lengths = Some(vec![200, 400]);
    assert!(
        !store
            .add_metrics([1.0, 2.0], f64::NAN, std::slice::from_ref(&with_lists))
            .unwrap()
    );
    assert_eq!(format!("{store:?}"), before);
    assert!(
        store
            .add_metrics([1.0, 2.0], 5.0, std::slice::from_ref(&with_lists))
            .unwrap()
    );
    assert_eq!(store.observation_count(), 1);
    assert!(!store.is_ready());
    // Prediction reads only the aggregate counters.
    assert_eq!(store.predict_metrics([1.0, 2.0], &scalar).unwrap(), None);
}
