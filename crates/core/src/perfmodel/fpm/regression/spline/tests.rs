// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use fit::{P, fit as batch_fit};

fn options(capacity: usize) -> ForwardPassPerfOptions {
    ForwardPassPerfOptions {
        max_observations: capacity,
        bucket_shape: Some([1, 1]),
        ..Default::default()
    }
}

fn periodic(step: usize) -> SplineFitConfig {
    SplineFitConfig {
        knots_per_axis: 2,
        search: SplineSearchConfig::Periodic { step },
    }
}

fn row(i: usize) -> Row {
    let x = ((i * 37) % 101) as f64 / 100.0;
    let z = ((i * 61) % 103) as f64 / 102.0;
    [
        x,
        z,
        3.0 + 100.0 * x.min(0.2)
            + 20.0 * (x - 0.2).max(0.0).min(0.4)
            + 2.0 * (x - 0.6).max(0.0)
            + 60.0 * z.min(0.3)
            + 15.0 * (z - 0.3).max(0.0).min(0.4)
            + (z - 0.7).max(0.0),
    ]
}

fn add(store: &mut BucketedSpline, value: Row) {
    assert!(store.add_observation([value[0], value[1]], value[2]));
}

#[test]
fn known_piecewise_surface_has_decreasing_nonnegative_slopes() {
    // Hand-derived continuous surface: slopes are 100/20/2 and 60/15/1,
    // so (x,z)=(.5,.5) is 3 + 20 + 6 + 18 + 3 = 50 ms.
    let knots = [vec![0.2, 0.6], vec![0.3, 0.7]];
    let truth = Model {
        knots: knots.clone(),
        slopes: [100.0, 20.0, 2.0, 60.0, 15.0, 1.0, 0.0, 0.0],
        intercept: 3.0,
        sse: 0.0,
    };
    let rows: Vec<Row> = (0..11)
        .flat_map(|x| (0..11).map(move |z| [x as f64 / 10.0, z as f64 / 10.0, 0.0]))
        .map(|mut r| {
            r[2] = 3.0
                + 100.0 * r[0].min(0.2)
                + 20.0 * (r[0] - 0.2).max(0.0).min(0.4)
                + 2.0 * (r[0] - 0.6).max(0.0)
                + 60.0 * r[1].min(0.3)
                + 15.0 * (r[1] - 0.3).max(0.0).min(0.4)
                + (r[1] - 0.7).max(0.0);
            r
        })
        .collect();
    let model = batch_fit(&rows, &knots).unwrap();
    assert!((model.predict([0.5, 0.5]) - 50.0).abs() < 1e-9);
    assert!(model.sse < 1e-17);
    for j in 0..6 {
        assert!((model.slopes[j] - truth.slopes[j]).abs() < 1e-8);
    }
    let affine = batch_fit(&rows, &[vec![], vec![]]).unwrap();
    let learned = fit::learn(&rows, 2, None).unwrap();
    assert!(learned.sse < affine.sse * 0.01);
    for a in 0..2 {
        assert!(learned.knots[a].windows(2).all(|w| w[0] < w[1]));
    }
}

#[test]
fn sparse_axes_and_boundary_models_are_finite_but_not_ready() {
    let rows: Vec<Row> = (0..40).map(|i| [0.0, (i % 3) as f64, 10.0]).collect();
    let model = fit::learn(&rows, 3, None).unwrap();
    assert_eq!(model.knots[0].len(), 0);
    assert_eq!(model.knots[1].len(), 2);
    assert!(!model.is_ready());
    assert_eq!(model.predict([3.0, 3.0]), 10.0);
    let decreasing: Vec<Row> = (0..40).map(|i| [i as f64, 0.0, 100.0 - i as f64]).collect();
    assert!(!fit::learn(&decreasing, 2, None).unwrap().is_ready());
}

fn assert_batch_parity(state: &mut RecursiveSpline, rows: &[Row], knots: &[Vec<f64>; 2]) {
    assert_eq!(state.count, rows.len());
    let recursive = state.fit_lazy(|| rows.to_vec());
    if rows.is_empty() {
        assert!(recursive.is_none());
        return;
    }
    let recursive = recursive.unwrap();
    let batch = batch_fit(rows, knots).unwrap();
    assert!(
        (recursive.sse - batch.sse).abs() < 1e-6 * batch.sse.max(1.0),
        "recursive SSE {} vs batch {}",
        recursive.sse,
        batch.sse
    );
    // Include off-span queries: nonunique retained-point fits can extrapolate
    // differently even when their training SSE agrees exactly.
    for x in [[0.0, 0.0], [0.1, 0.9], [1.0, 1.0], [1.5, 0.0]] {
        let actual = recursive.predict(x);
        let expected = batch.predict(x);
        assert!(
            (actual - expected).abs() < 2e-6 * expected.abs().max(1.0),
            "{x:?}: {actual} vs {expected}"
        );
    }
    assert!(recursive.slopes.iter().all(|v| *v >= 0.0));
}

#[test]
fn recursive_fit_matches_batch_through_eviction_rank_loss_and_duplicates() {
    let knots = [vec![0.2, 0.6], vec![0.3, 0.7]];
    let mut state = RecursiveSpline::new(&[], &knots, None);
    let mut rows = VecDeque::new();
    for i in 0..2400 {
        let mut r = row(i);
        if i % 600 < 200 {
            r[1] = r[0];
            r[2] = 5.0 + 20.0 * r[0];
        } else if i % 600 < 400 {
            r[1] = 0.3;
        }
        if i % 71 == 0 {
            r = [0.1, 0.2, 25.0];
        }
        state.add(r);
        rows.push_back(r);
        if rows.len() > 64 {
            state.remove(rows.pop_front().unwrap());
        }
        if i % 11 == 0 {
            assert_batch_parity(&mut state, rows.make_contiguous(), &knots);
        }
    }
    assert!(state.batch_fallbacks > 0);
    while let Some(r) = rows.pop_front() {
        state.remove(r);
        assert_batch_parity(&mut state, rows.make_contiguous(), &knots);
    }
    state.add([0.1, 0.9, 32.0]);
    assert_batch_parity(&mut state, &[[0.1, 0.9, 32.0]], &knots);
}

#[test]
fn numerical_rebuild_and_configured_mutation_clocks_are_independent() {
    let knots = [vec![0.2, 0.6], vec![0.3, 0.7]];
    let mut rows: Vec<Row> = (0..64).map(row).collect();
    let mut state = RecursiveSpline::new(&rows, &knots, Some(3));
    let old = rows.remove(0);
    let next = row(100);
    rows.push(next);
    state.add(next);
    state.remove(old);
    state.fit_lazy(|| rows.clone()).unwrap();
    assert_eq!(state.mutations, 2);
    assert_eq!(state.rebuilds, 0);
    let old = rows.remove(0);
    let next = row(101);
    rows.push(next);
    state.add(next);
    state.remove(old);
    state.fit_lazy(|| rows.clone()).unwrap();
    assert_eq!(state.mutations, 0);
    assert_eq!(state.rebuilds, 1); // threshold crossed by the entire insert/evict transaction
    let mut disabled = RecursiveSpline::new(&rows, &knots, None);
    disabled.damage_for_test();
    // Numerical recovery still runs when periodic moment reconstruction is off.
    assert_batch_parity(&mut disabled, &rows, &knots);
    assert_eq!(disabled.rebuilds, 1);
}

#[test]
fn startup_periods_and_minimum_use_accepted_count_not_retained_count() {
    for step in [8, 16, 32, 64, 128] {
        let mut store = BucketedSpline::new(&options(32), periodic(step), None);
        for i in 1..=130 {
            add(&mut store, row(i));
            let expected = if i < 32 { 0 } else { 1 + i / step - 32 / step };
            assert_eq!(store.searches, expected as u64, "step={step}, accepted={i}");
            assert_eq!(store.observation_count(), i.min(32));
        }
    }
    let mut opts = options(64);
    opts.min_observations = 40;
    let mut store = BucketedSpline::new(&opts, periodic(64), None);
    for i in 0..39 {
        add(&mut store, row(i));
    }
    assert!(!store.diagnostics().initialized);
    add(&mut store, row(39));
    assert!(store.diagnostics().initialized);
    assert_eq!(store.last_search, Some(40));
}

#[test]
fn constant_initialization_does_not_repeat_search_or_claim_readiness() {
    let mut store = BucketedSpline::new(&options(64), periodic(10000), None);
    for _ in 0..40 {
        add(&mut store, [1.0, 1.0, 10.0]);
    }
    assert!(store.diagnostics().initialized);
    assert!(!store.diagnostics().ready);
    assert!(!store.is_ready());
    assert_eq!(store.searches, 1);
    assert!(store.predict(&[1.0, 1.0]).is_none());
    for i in 1..12 {
        add(&mut store, [i as f64, 1.0, 10.0 + i as f64]);
    }
    assert!(store.diagnostics().ready);
    assert_eq!(store.searches, 1);
}

#[test]
fn rolling_monitor_evicts_clears_and_does_not_require_current_error() {
    let mut monitor = ErrorMonitor::default();
    for bad in [true, true, false, false, true] {
        monitor.observe(bad, 3);
    }
    assert_eq!(monitor.bad, 1);
    assert_eq!(monitor.errors.len(), 3);
    monitor.clear();
    assert_eq!(monitor.bad, 0);
    assert!(monitor.errors.is_empty());
    let mut store = BucketedSpline::new(
        &options(64),
        SplineFitConfig {
            knots_per_axis: 2,
            search: SplineSearchConfig::Adaptive {
                window: 16,
                trigger: 8,
                tolerance: 0.05,
                absolute_tolerance_ms: 1.0,
                cooldown: 64,
            },
        },
        None,
    );
    for i in 0..32 {
        add(&mut store, row(i));
    }
    store.accepted = 95;
    for _ in 0..8 {
        store.monitor.observe(true, 16);
    }
    assert!(!store.search_due());
    store.accepted = 96;
    store.monitor.observe(false, 16);
    assert!(store.search_due());
    store.search();
    assert_eq!(store.last_search, Some(96));
    assert!(store.monitor.errors.is_empty());
    assert!(!store.search_due());
}

#[test]
fn adaptive_error_uses_raw_preupdate_prediction_even_when_guard_falls_back() {
    let mut store = BucketedSpline::new(
        &options(64),
        SplineFitConfig {
            knots_per_axis: 2,
            search: SplineSearchConfig::Adaptive {
                window: 16,
                trigger: 8,
                tolerance: 0.05,
                absolute_tolerance_ms: 1.0,
                cooldown: 64,
            },
        },
        None,
    );
    for i in 0..32 {
        add(&mut store, row(i));
    }
    let x = [5.0, 7.0];
    assert!(!store.inside(&x));
    let prior = store.raw_spline_prediction(&x).unwrap();
    add(&mut store, [x[0], x[1], prior + 100.0]);
    assert_eq!(store.monitor.errors.back(), Some(&true));
    assert_eq!(store.monitor.bad, 1);
    assert_eq!(store.last_search, Some(32));
    let before = store.diagnostics();
    for _ in 0..5 {
        let _ = store.predict(&x);
    }
    assert_eq!(store.diagnostics(), before);
    // A one-ms residual is exactly the floor and must not count as a bad error.
    store.spline_fit = Some(Model {
        knots: [vec![], vec![]],
        slopes: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        intercept: 9.0,
        sse: 0.0,
    });
    store.scale = [1.0; 2];
    add(&mut store, [0.0, 0.0, 10.0]);
    assert_eq!(store.monitor.errors.back(), Some(&false));
}

#[test]
fn rejection_is_atomic_and_guard_tracks_retirement_not_historical_bounds() {
    let mut store = BucketedSpline::new(&options(32), periodic(64), Some(7));
    for i in 0..32 {
        add(&mut store, [i as f64, 0.0, 10.0 + i as f64]);
    }
    assert!(store.inside(&[0.0, 0.0]));
    add(&mut store, [32.0, 0.0, 42.0]); // One bucket deterministically evicts the oldest.
    assert!(!store.inside(&[0.0, 0.0]));
    assert_eq!(store.minimum, [1.0, 0.0]);
    assert_eq!(store.maximum, [32.0, 0.0]);
    let expected = store
        .linear_fit
        .as_ref()
        .unwrap()
        .predict(&[0.0, 0.0])
        .unwrap()
        .max(1e-6);
    assert_eq!(store.predict(&[0.0, 0.0]), Some(expected));
    let before = store.diagnostics();
    let mutations = store.mutations_since_rebuild();
    for (x, y) in [
        ([f64::NAN, 0.0], 1.0),
        ([-1.0, 0.0], 1.0),
        ([1.0, 1.0], f64::INFINITY),
        ([1.0, 1.0], 0.0),
    ] {
        assert!(!store.add_observation(x, y));
    }
    assert_eq!(store.diagnostics(), before);
    assert_eq!(store.mutations_since_rebuild(), mutations);
}

#[test]
fn fallback_linear_uses_the_exact_shared_retained_set() {
    let mut opts = options(64);
    opts.bucket_shape = Some([4, 4]);
    let mut store = BucketedSpline::new(&opts, periodic(128), None);
    for i in 0..180 {
        add(&mut store, row(i));
    }
    let retained: Vec<_> = store
        .samples
        .observations()
        .into_iter()
        .map(|(_, entry)| entry.observation)
        .collect();
    let mut linear = RecursiveFit::new(opts.regression_ridge_scale, None);
    for &r in &retained {
        linear.add(r);
    }
    let reference = linear
        .fit_lazy(opts.min_observations, || retained.clone())
        .unwrap();
    for x in [[2.0, 3.0], [4.0, 0.0]] {
        let expected = reference.predict(&x).unwrap().max(1e-6);
        assert!((store.predict(&x).unwrap() - expected).abs() < 1e-8);
    }
}

#[test]
fn finite_extreme_targets_fail_closed_and_large_offsets_keep_constant_axes_inactive() {
    let knots = [vec![0.2, 0.6], vec![0.3, 0.7]];
    let rows: Vec<Row> = (0..64)
        .map(|i| [1e12, 1e6 + (i % 7) as f64 * 1e-6, 72000.0 + (i % 7) as f64])
        .collect();
    let mut state = RecursiveSpline::new(&rows, &knots, None);
    assert_batch_parity(&mut state, &rows, &knots);
    let huge = [[0.0, 0.0, f64::MAX], [1.0, 1.0, 1.0]];
    assert!(batch_fit(&huge, &knots).is_none());
    let mut state = RecursiveSpline::new(&huge, &knots, None);
    assert!(state.fit_lazy(|| huge.to_vec()).is_none());
    assert_eq!(P, 8);
}

#[test]
fn negative_raw_prediction_is_monitored_before_serving_floor() {
    let mut store = BucketedSpline::new(
        &options(64),
        SplineFitConfig {
            knots_per_axis: 2,
            search: SplineSearchConfig::Adaptive {
                window: 16,
                trigger: 8,
                tolerance: 0.05,
                absolute_tolerance_ms: 1.0,
                cooldown: 64,
            },
        },
        None,
    );
    for i in 0..32 {
        add(&mut store, row(i));
    }
    store.spline_fit = Some(Model {
        knots: [vec![], vec![]],
        slopes: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        intercept: -10.0,
        sse: 0.0,
    });
    store.scale = [1.0; 2];
    assert_eq!(store.raw_spline_prediction(&[0.0, 0.0]), Some(-10.0));
    // Flooring -10 to 1e-6 would falsely classify the error against .5 as good.
    add(&mut store, [0.0, 0.0, 0.5]);
    assert_eq!(store.monitor.errors.back(), Some(&true));
}

#[test]
fn newly_inserted_then_immediately_retired_point_does_not_expand_guard() {
    let mut store = BucketedSpline::new(&options(32), periodic(64), None);
    for i in 0..32 {
        add(&mut store, row(i));
    }
    let minimum = store.minimum;
    let maximum = store.maximum;
    let incoming = RegressionObservation {
        raw_x: [100.0, 200.0],
        observed_ms: 500.0,
    };
    // End state of a capacity-limited sampler transaction that retires its own
    // incoming point: retained samples are unchanged, and the actual eviction
    // is exactly the incoming payload. Tied one-element buckets can do this.
    store.update_bounds(incoming, Some(incoming));
    assert_eq!(store.minimum, minimum);
    assert_eq!(store.maximum, maximum);
    assert!(!store.inside(&incoming.raw_x));
}

#[test]
fn adaptive_boundary_fit_monitors_error_without_claiming_readiness() {
    let mut store = BucketedSpline::new(
        &options(64),
        SplineFitConfig {
            knots_per_axis: 2,
            search: SplineSearchConfig::Adaptive {
                window: 16,
                trigger: 8,
                tolerance: 0.05,
                absolute_tolerance_ms: 1.0,
                cooldown: 64,
            },
        },
        None,
    );
    for _ in 0..32 {
        add(&mut store, [1.0, 1.0, 10.0]);
    }
    assert!(store.diagnostics().initialized);
    assert!(!store.diagnostics().ready);
    assert_eq!(store.raw_spline_prediction(&[1.0, 1.0]), Some(10.0));
    assert_eq!(store.predict(&[1.0, 1.0]), None);
    for i in 33..=96 {
        add(&mut store, [1.0, 1.0, if i % 2 == 0 { 20.0 } else { 40.0 }]);
        assert!(!store.diagnostics().ready);
        assert_eq!(store.searches, if i < 96 { 1 } else { 2 });
    }
    assert_eq!(store.last_search, Some(96));
    assert!(store.monitor.errors.is_empty());
}

#[test]
fn absent_prediction_recovers_coefficients_without_inventing_search_events() {
    let mut store = BucketedSpline::new(
        &options(64),
        SplineFitConfig {
            knots_per_axis: 2,
            search: SplineSearchConfig::Adaptive {
                window: 16,
                trigger: 8,
                tolerance: 0.05,
                absolute_tolerance_ms: 1.0,
                cooldown: 64,
            },
        },
        None,
    );
    for i in 0..32 {
        add(&mut store, row(i));
    }
    store.spline_fit = None;
    assert!(store.monitor.errors.is_empty());
    add(&mut store, row(32));
    assert!(store.spline_fit.is_some());
    assert!(store.monitor.errors.is_empty());
    assert_eq!(store.searches, 1);
    assert_eq!(store.last_search, Some(32));
}

#[test]
fn knot_relocation_preserves_the_linear_rebuild_mutation_clock() {
    let mut store = BucketedSpline::new(&options(32), periodic(64), Some(70));
    for i in 0..32 {
        add(&mut store, row(i));
    }
    assert_eq!(store.searches, 1);
    assert_eq!(store.mutations_since_rebuild(), 32);
    assert_eq!(store.recursive.as_ref().unwrap().mutations, 0);
    let initial_scale = store.scale;
    for i in 32..63 {
        let mut sample = row(i);
        sample[0] *= 10.0;
        sample[1] *= 5.0;
        add(&mut store, sample);
    }
    // Each accepted sample after filling capacity contributes an insertion and
    // an actual eviction. The linear interval of 70 was reached at accepted 51:
    // 32 + 19 * 2 = 70. The next 12 transactions leave its clock at 24.
    assert_eq!(store.accepted, 63);
    assert_eq!(store.observation_count(), 32);
    assert_eq!(store.mutations_since_rebuild(), 24);
    let old_knots = store.spline_fit.as_ref().unwrap().knots.clone();
    let mut sample = row(63);
    sample[0] *= 10.0;
    sample[1] *= 5.0;
    add(&mut store, sample);
    assert_eq!(store.accepted, 64);
    assert_eq!(store.observation_count(), 32);
    assert_eq!(store.searches, 2);
    assert_eq!(store.last_search, Some(64));
    assert_ne!(store.scale, initial_scale);
    assert_ne!(store.spline_fit.as_ref().unwrap().knots, old_knots);
    // Searching after this complete insertion/eviction rebuilds the spline
    // epoch only. It must not reset or otherwise alter the linear clock.
    assert_eq!(store.recursive.as_ref().unwrap().mutations, 0);
    assert_eq!(store.mutations_since_rebuild(), 26);
}

#[test]
fn three_knots_per_axis_fit_an_independent_eight_slope_oracle() {
    let knots = [vec![0.2, 0.5, 0.8], vec![0.1, 0.4, 0.7]];
    let rows: Vec<Row> = (0..11)
        .flat_map(|i| (0..11).map(move |j| (i as f64 / 10.0, j as f64 / 10.0)))
        .map(|(x, z)| {
            // Explicit scalar surface, independent of features/Model::predict.
            let y = 3.0
                + 100.0 * x.min(0.2)
                + 40.0 * (x - 0.2).max(0.0).min(0.3)
                + 10.0 * (x - 0.5).max(0.0).min(0.3)
                + 2.0 * (x - 0.8).max(0.0)
                + 60.0 * z.min(0.1)
                + 20.0 * (z - 0.1).max(0.0).min(0.3)
                + 5.0 * (z - 0.4).max(0.0).min(0.3)
                + (z - 0.7).max(0.0);
            [x, z, y]
        })
        .collect();
    let model = batch_fit(&rows, &knots).unwrap();
    assert!(model.sse < 1e-17);
    assert!((model.intercept - 3.0).abs() < 1e-9);
    let expected_slopes = [100.0, 40.0, 10.0, 2.0, 60.0, 20.0, 5.0, 1.0];
    for (actual, expected) in model.slopes.iter().zip(expected_slopes) {
        assert!((actual - expected).abs() < 1e-8);
    }
    // At (.6,.6), the two axis contributions are 33 and 13 ms: total 49 ms.
    assert!((model.predict([0.6, 0.6]) - 49.0).abs() < 1e-9);
    let mut recursive = RecursiveSpline::new(&rows, &knots, None);
    assert_batch_parity(&mut recursive, &rows, &knots);
    let affine = batch_fit(&rows, &[vec![], vec![]]).unwrap();
    let learned = fit::learn(&rows, 3, None).unwrap();
    assert_eq!(learned.knots[0].len(), 3);
    assert_eq!(learned.knots[1].len(), 3);
    assert!(learned.sse < affine.sse * 0.01);
}
