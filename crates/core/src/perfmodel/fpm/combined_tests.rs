// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{BackendKind, ScheduledRequestMetrics};
use parquet::data_type::{DoubleType, Int64Type};
use parquet::file::{properties::WriterProperties, writer::SerializedFileWriter};
use parquet::schema::parser::parse_message_type;
use sha2::{Digest, Sha256};
use std::path::Path;

type Json = serde_json::Value;

fn hash(path: &Path) -> String {
    format!("{:x}", Sha256::digest(std::fs::read(path).unwrap()))
}

// A constant 10 ms forest exactly fitted to two 10 ms historical rows.
// This is a real training checkpoint, so tests exercise canonical validation,
// payload hashes, feature projection, fitting and publication together.
fn checkpoint(role: ForwardPassWorkerType) -> (tempfile::TempDir, ForwardPassPerfModelConfig) {
    checkpoint_with_forest(role, 10.0, &[0.0])
}

fn checkpoint_with_forest(
    role: ForwardPassWorkerType,
    baseline: f64,
    tree_values: &[f64],
) -> (tempfile::TempDir, ForwardPassPerfModelConfig) {
    let dir = tempfile::tempdir().unwrap();
    let mut config =
        ForwardPassPerfModelConfig::new("selector-test", "test-gpu", BackendKind::Vllm, role);
    config.estimation_mode = EstimationMode::FpmLearned;
    config.estimator_config.fpm_learned.adaptation =
        super::super::super::LearnedAdaptation::CausalSelector;
    config.estimator_config.fpm_learned.checkpoint_path = Some(dir.path().to_owned());
    config.estimator_config.fpm_learned.update_mode =
        super::super::super::LearnedUpdateMode::LeafOnly;
    config.estimator_config.fpm_learned.update_every = 1;
    config.estimator_config.fpm_learned.recent_window = 8;
    config.estimator_config.correction.min_observations = 1;
    config.estimator_config.correction.sampling.bins_per_axis = [2, 2];
    config.estimator_config.selector.alpha = 0.4;
    config.estimator_config.selector.warmup = 3;
    config.estimator_config.selector.margin = 0.025;
    config.estimator_config.selector.persistence = 2;
    let rows = dir.path().join("rows.parquet");
    let schema = Arc::new(parse_message_type("message seed { REQUIRED INT64 row_id (UINT_64); REQUIRED DOUBLE y; REQUIRED DOUBLE wall_time_ms; REQUIRED DOUBLE x0; }").unwrap());
    let mut writer = SerializedFileWriter::new(
        std::fs::File::create(&rows).unwrap(),
        schema,
        Arc::new(WriterProperties::builder().build()),
    )
    .unwrap();
    let mut group = writer.next_row_group().unwrap();
    let mut column = group.next_column().unwrap().unwrap();
    column
        .typed::<Int64Type>()
        .write_batch(&[0, 1], None, None)
        .unwrap();
    column.close().unwrap();
    for values in [[10.0, 10.0], [10.0, 10.0], [0.0, 1.0]] {
        let mut column = group.next_column().unwrap().unwrap();
        column
            .typed::<DoubleType>()
            .write_batch(&values, None, None)
            .unwrap();
        column.close().unwrap();
    }
    group.close().unwrap();
    writer.close().unwrap();
    let trees: Vec<_> = tree_values
        .iter()
        .map(|value| {
            serde_json::json!({
                "left": [-1], "right": [-1], "feature": [-1], "threshold": [0.0],
                "value": [value], "missing_left": [true]
            })
        })
        .collect();
    let store = serde_json::json!({"baseline": baseline, "trees": trees});
    let seed_store = serde_json::json!({
        "rows_file": "rows.parquet", "rows_sha256": hash(&rows), "row_count": 2,
        "bin_thresholds": [[0.5]], "n_bins_non_missing": [2], "missing_values_bin_idx": 255,
        "loss": "squared_error", "learning_rate": 1.0, "l2_regularization": 0.0,
        "min_samples_leaf": 1, "max_depth": null, "max_leaf_nodes": 2
    });
    let mut stores = serde_json::Map::new();
    let mut seeds = serde_json::Map::new();
    for kind in workload_kinds_for(role) {
        let name = serde_json::to_value(kind)
            .unwrap()
            .as_str()
            .unwrap()
            .to_owned();
        stores.insert(name.clone(), store.clone());
        seeds.insert(name, seed_store.clone());
    }
    let identity = canonical_learned_identity(&config).unwrap();
    let artifact = serde_json::json!({
        "schema": "aic_fpm_learned_forward_perf", "schema_version": 1,
        "worker_type": role, "target": "ms", "features": ["num_decode_requests"],
        "stores": stores, "metadata": {"identity": identity}
    });
    let artifact_path = dir.path().join("model.json");
    std::fs::write(&artifact_path, artifact.to_string()).unwrap();
    let manifest = serde_json::json!({
        "schema": "aisimulate_fpm_training_checkpoint", "schema_version": 1,
        "artifact_file": "model.json", "artifact_sha256": hash(&artifact_path),
        "identity": identity, "features": ["num_decode_requests"], "target": "ms",
        "shrinkage": "included_in_leaf_values", "stores": seeds
    });
    std::fs::write(dir.path().join("manifest.json"), manifest.to_string()).unwrap();
    (dir, config)
}

fn metric(prefill: u32, decode: u32, kv: u32, wall_ms: f64) -> ForwardPassMetrics {
    ForwardPassMetrics {
        wall_time: wall_ms / 1000.0,
        scheduled_requests: ScheduledRequestMetrics {
            num_prefill_requests: u32::from(prefill != 0),
            sum_prefill_tokens: prefill,
            num_decode_requests: decode,
            sum_decode_kv_tokens: kv,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn estimate(model: &ForwardPassPerfModel, metrics: &[ForwardPassMetrics]) -> f64 {
    model
        .estimate_forward_pass_time_ms(metrics)
        .unwrap()
        .unwrap()
}

fn online_estimate(model: &ForwardPassPerfModel, metrics: &[ForwardPassMetrics]) -> f64 {
    let ForwardPassPerfMode::CausalSelector(model) = &model.mode else {
        panic!("expected combined model")
    };
    let Some((kind, Some(features))) = prepare_learned_iteration(&model.learned, metrics).unwrap()
    else {
        panic!("expected supported nonempty query")
    };
    model.learned.predict_ms(kind, &features).unwrap().unwrap()
}

fn state(model: &ForwardPassPerfModel, store: &str, query: &str) -> Json {
    model.learned_selector_diagnostics().unwrap()["states"]
        .as_array()
        .unwrap()
        .iter()
        .find(|value| value["store"] == store && value["query_type"] == query)
        .unwrap()
        .clone()
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-10 * actual.abs().max(expected.abs()).max(1.0),
        "{actual} != {expected}"
    );
}

#[test]
fn combined_trajectory_matches_independent_experts_and_scalar_gate() {
    let (_dir, config) = checkpoint(ForwardPassWorkerType::Decode);
    let mut stable_config = config.clone();
    stable_config.estimator_config.fpm_learned.adaptation =
        super::super::super::LearnedAdaptation::Frozen;
    stable_config.estimator_config.selector = Default::default();
    let mut online_config = config.clone();
    online_config.estimator_config.fpm_learned.adaptation =
        super::super::super::LearnedAdaptation::Online;
    online_config.estimator_config.correction.enabled = false;
    online_config.estimator_config.selector = Default::default();
    let mut stable = ForwardPassPerfModel::best_available(stable_config).unwrap();
    let mut online = ForwardPassPerfModel::best_available(online_config).unwrap();
    let mut combined = ForwardPassPerfModel::best_available(config.clone()).unwrap();
    let mut active = 0;
    let mut ewma = [0.0; 2];
    let mut streak = 0;
    let mut switches = 0;
    let targets = [
        30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 5.0, 5.0, 5.0, 5.0, 5.0, 15.0, 8.0,
    ];
    for (index, target) in targets.into_iter().enumerate() {
        let observation = vec![metric(0, 2, 1000, target)];
        let predictions = [
            estimate(&stable, &observation),
            estimate(&online, &observation),
        ];
        let before = combined.learned_selector_diagnostics();
        close(estimate(&combined, &observation), predictions[active]);
        close(
            estimate(&combined, &[metric(0, 2, 1000, f64::NAN)]),
            predictions[active],
        );
        assert_eq!(combined.learned_selector_diagnostics(), before);
        for expert in 0..2 {
            let loss = (predictions[expert] / target - 1.0).abs();
            ewma[expert] = if index == 0 {
                loss
            } else {
                0.6 * ewma[expert] + 0.4 * loss
            };
        }
        let challenger = 1 - active;
        streak = if index + 1 >= 3 && ewma[challenger] < 0.975 * ewma[active] {
            streak + 1
        } else {
            0
        };
        if streak >= 2 {
            active = challenger;
            streak = 0;
            switches += 1;
        }
        stable.tune_with_fpms(&[observation.clone()]).unwrap();
        online.tune_with_fpms(&[observation.clone()]).unwrap();
        combined.tune_with_fpms(&[observation]).unwrap();
        let actual = state(&combined, "pure_decode", "decode");
        assert_eq!(actual["observations"], index + 1);
        assert_eq!(
            actual["selected"],
            if active == 0 { "stable" } else { "online" }
        );
        close(actual["stable_ewma"].as_f64().unwrap(), ewma[0]);
        close(actual["online_ewma"].as_f64().unwrap(), ewma[1]);
        assert_eq!(actual["switches"], switches);
    }
    assert!(switches > 0);
    let ForwardPassPerfMode::CausalSelector(model) = &combined.mode else {
        panic!()
    };
    assert_eq!(
        model
            .stable
            .store(ForwardPassRegressionWorkloadKind::PureDecode)
            .unwrap()
            .baseline,
        10.0
    );
    assert_eq!(
        model
            .stable
            .store(ForwardPassRegressionWorkloadKind::PureDecode)
            .unwrap()
            .trees[0]
            .value,
        [0.0]
    );
    let saved = combined.provenance().unwrap().config.clone();
    assert_eq!(
        serde_json::from_str::<ForwardPassPerfModelConfig>(&serde_json::to_string(&saved).unwrap())
            .unwrap(),
        saved
    );
}

#[test]
fn combined_calibration_and_selector_do_not_bleed_across_stores_queries_or_clones() {
    let (_dir, mut config) = checkpoint(ForwardPassWorkerType::Aggregated);
    config.estimator_config.selector.warmup = 100;
    let mut combined = ForwardPassPerfModel::best_available(config).unwrap();
    let untouched = combined.clone();
    let local = vec![metric(16, 1, 100, 20.0), metric(0, 2, 5000, 20.0)];
    let cross = vec![metric(16, 0, 0, 20.0), metric(0, 2, 5000, 20.0)];
    let mixed_query = vec![metric(16, 1, 100, 20.0)];
    for _ in 0..5 {
        combined
            .tune_with_fpms(std::slice::from_ref(&local))
            .unwrap();
    }
    close(estimate(&combined, &local), 20.0);
    close(estimate(&combined, &cross), 10.0);
    close(estimate(&combined, &mixed_query), 10.0);
    close(estimate(&untouched, &local), 10.0);
    assert_eq!(
        state(&combined, "contains_locally_mixed", "decode")["observations"],
        5
    );
    assert_eq!(
        state(&combined, "cross_rank_aggregated", "decode")["observations"],
        0
    );
    assert_eq!(
        state(&combined, "contains_locally_mixed", "mixed")["observations"],
        0
    );
    assert_eq!(
        state(&untouched, "contains_locally_mixed", "decode")["observations"],
        0
    );
    let diagnostics = combined.learned_selector_diagnostics().unwrap();
    let calibration = diagnostics["calibration"].as_array().unwrap();
    assert_eq!(
        calibration
            .iter()
            .filter(|item| item["retained_observations"].as_u64().unwrap() > 0)
            .count(),
        1
    );
}

#[test]
fn combined_fit_failure_admits_once_and_rejected_retry_or_flush_never_recounts() {
    let (_dir, mut config) = checkpoint(ForwardPassWorkerType::Decode);
    config.estimator_config.selector.alpha = 1.0;
    config.estimator_config.selector.warmup = 1;
    let mut combined = ForwardPassPerfModel::best_available(config).unwrap();
    let extreme = vec![metric(0, 2, 1000, 1.0e308)];
    combined
        .tune_with_fpms(std::slice::from_ref(&extreme))
        .unwrap();
    let error = combined.tune_with_fpms(&[extreme]).unwrap_err();
    assert!(
        error.to_string().contains("feedback was admitted"),
        "{error}"
    );
    let evidence = combined.learned_selector_diagnostics();
    assert_eq!(state(&combined, "pure_decode", "decode")["observations"], 2);
    let online = combined.learned_online_diagnostics().unwrap();
    assert_eq!(online["stores"]["pure_decode"]["accepted_observations"], 2);
    assert_eq!(online["stores"]["pure_decode"]["pending_rows"], 1);
    assert!(
        combined
            .tune_with_fpms(&[vec![metric(0, 2, 1000, 20.0)]])
            .is_err()
    );
    assert_eq!(combined.learned_selector_diagnostics(), evidence);
    assert!(combined.flush_updates().is_err());
    assert_eq!(combined.learned_selector_diagnostics(), evidence);
}

#[test]
fn combined_global_age_flush_expires_online_rows_without_recounting_selector_evidence() {
    for refresh in [
        super::super::super::LearnedDerivativeRefresh::Eager,
        super::super::super::LearnedDerivativeRefresh::Lazy,
    ] {
        // One row is pending only; two are fitted only; three leave both.
        for prefill_count in 1..=3 {
            let (_directory, mut config) = checkpoint(ForwardPassWorkerType::Aggregated);
            let controls = &mut config.estimator_config.fpm_learned;
            controls.derivative_refresh = refresh;
            controls.update_every = 2;
            controls.recent_max_age_observations = Some(3);
            controls.history_weight = Some(0.0);
            let mut combined = ForwardPassPerfModel::best_available(config).unwrap();
            let prefill = vec![metric(16, 0, 0, 30.0)];
            for _ in 0..prefill_count {
                combined
                    .tune_with_fpms(std::slice::from_ref(&prefill))
                    .unwrap();
            }
            for _ in 0..6 {
                combined
                    .tune_with_fpms(&[vec![metric(0, 2, 1000, 50.0)]])
                    .unwrap();
            }
            close(
                online_estimate(&combined, &prefill),
                if prefill_count == 1 { 10.0 } else { 30.0 },
            );
            let before = combined.learned_selector_diagnostics();
            assert_eq!(
                state(&combined, "pure_prefill", "prefill")["observations"],
                prefill_count
            );
            combined.flush_updates().unwrap();
            // Only online retained rows expire. Calibration and the causal
            // record of earlier admissions are independent state.
            assert_eq!(combined.learned_selector_diagnostics(), before);
            close(online_estimate(&combined, &prefill), 10.0);
            let diagnostics = combined.learned_online_diagnostics().unwrap();
            let store = &diagnostics["stores"]["pure_prefill"];
            assert_eq!(store["retained_row_ids"], serde_json::json!([]));
            assert_eq!(store["recent_rows"], 0);
            assert_eq!(store["pending_rows"], 0);
            assert_eq!(store["accepted_observations"], prefill_count);
            assert_eq!(store["evicted_observations"], prefill_count);
            assert_eq!(
                store["publications"],
                if prefill_count == 1 { 0 } else { 2 }
            );
            assert_eq!(store["history_row_weight"], 1.0);
            combined.flush_updates().unwrap();
            assert_eq!(combined.learned_online_diagnostics().unwrap(), diagnostics);
            assert_eq!(combined.learned_selector_diagnostics(), before);
        }
    }
}

#[test]
fn combined_rejected_and_idle_feedback_preserves_global_age_boundary() {
    for refresh in [
        super::super::super::LearnedDerivativeRefresh::Eager,
        super::super::super::LearnedDerivativeRefresh::Lazy,
    ] {
        let (_directory, mut config) = checkpoint(ForwardPassWorkerType::Aggregated);
        let controls = &mut config.estimator_config.fpm_learned;
        controls.derivative_refresh = refresh;
        controls.update_every = 2;
        controls.recent_max_age_observations = Some(3);
        controls.history_weight = Some(0.0);
        let mut combined = ForwardPassPerfModel::best_available(config).unwrap();
        combined
            .tune_with_fpms(&[vec![metric(16, 0, 0, 20.0)]])
            .unwrap(); // ID 2.
        combined
            .tune_with_fpms(&[vec![metric(16, 0, 0, 40.0)]])
            .unwrap(); // ID 3.
        combined
            .tune_with_fpms(&[vec![metric(0, 2, 1000, 50.0)]])
            .unwrap(); // ID 4.
        let evidence = combined.learned_selector_diagnostics();
        for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            combined
                .tune_with_fpms(&[vec![metric(0, 2, 1000, invalid)]])
                .unwrap();
        }
        combined
            .tune_with_fpms(&[vec![ForwardPassMetrics::default()]])
            .unwrap();
        assert!(combined.tune_with_fpms(&[vec![]]).is_err());
        assert_eq!(combined.learned_selector_diagnostics(), evidence);
        combined.flush_updates().unwrap();
        assert_eq!(combined.learned_selector_diagnostics(), evidence);
        let diagnostics = combined.learned_online_diagnostics().unwrap();
        assert_eq!(diagnostics["rejected_observations"], 5);
        assert_eq!(
            diagnostics["stores"]["pure_prefill"]["retained_row_ids"],
            serde_json::json!([2, 3])
        );
        close(online_estimate(&combined, &[metric(16, 0, 0, 0.0)]), 30.0);
        combined
            .tune_with_fpms(&[vec![metric(0, 2, 1000, 60.0)]])
            .unwrap(); // ID 5.
        let evidence = combined.learned_selector_diagnostics();
        combined.flush_updates().unwrap();
        assert_eq!(combined.learned_selector_diagnostics(), evidence);
        assert_eq!(
            combined.learned_online_diagnostics().unwrap()["stores"]["pure_prefill"]["retained_row_ids"],
            serde_json::json!([3])
        );
        close(online_estimate(&combined, &[metric(16, 0, 0, 0.0)]), 40.0);
    }
}

#[test]
fn combined_restricted_tail_uses_frozen_prefix_with_independent_online_predictions() {
    for refresh in [
        super::super::super::LearnedDerivativeRefresh::Eager,
        super::super::super::LearnedDerivativeRefresh::Lazy,
    ] {
        for grouping in [false, true] {
            // A nonzero immutable tree supplies the whole 10 ms seed; only the
            // final zero-valued tree can change under online feedback.
            let (_directory, mut config) =
                checkpoint_with_forest(ForwardPassWorkerType::Decode, 0.0, &[10.0, 0.0]);
            let controls = &mut config.estimator_config.fpm_learned;
            controls.derivative_refresh = refresh;
            controls.history_bin_grouping = grouping;
            controls.mutable_tree_tail = Some(1);
            let mut online_config = config.clone();
            online_config.estimator_config.fpm_learned.adaptation =
                super::super::super::LearnedAdaptation::Online;
            online_config.estimator_config.correction.enabled = false;
            online_config.estimator_config.selector = Default::default();
            let mut online = ForwardPassPerfModel::best_available(online_config).unwrap();
            let mut combined = ForwardPassPerfModel::best_available(config).unwrap();
            for label in [30.0, 40.0, 5.0, 50.0] {
                let observation = vec![metric(0, 2, 1000, label)];
                close(
                    online_estimate(&combined, &observation),
                    estimate(&online, &observation),
                );
                online
                    .tune_with_fpms(std::slice::from_ref(&observation))
                    .unwrap();
                combined
                    .tune_with_fpms(std::slice::from_ref(&observation))
                    .unwrap();
                close(
                    online_estimate(&combined, &observation),
                    estimate(&online, &observation),
                );
            }
            let ForwardPassPerfMode::CausalSelector(model) = &combined.mode else {
                panic!("expected combined model")
            };
            let store = model
                .learned
                .store(ForwardPassRegressionWorkloadKind::PureDecode)
                .unwrap();
            assert_eq!(store.trees[0].value, [10.0]);
            assert_ne!(store.trees[1].value, [0.0]);
            assert_eq!(
                model
                    .stable
                    .store(ForwardPassRegressionWorkloadKind::PureDecode)
                    .unwrap()
                    .trees[1]
                    .value,
                [0.0]
            );
        }
    }
}

#[test]
fn combined_persistence_apis_reject_partial_state_and_standalone_online_resume_stays_valid() {
    let (dir, config) = checkpoint(ForwardPassWorkerType::Decode);
    let combined = ForwardPassPerfModel::best_available(config.clone()).unwrap();
    assert!(
        combined
            .export_learned_artifact()
            .unwrap_err()
            .to_string()
            .contains("combined artifact export")
    );
    assert!(
        combined
            .save_online_checkpoint(dir.path().join("combined"))
            .unwrap_err()
            .to_string()
            .contains("combined checkpoint saving")
    );
    let mut online_config = config.clone();
    online_config.estimator_config.fpm_learned.adaptation =
        super::super::super::LearnedAdaptation::Online;
    online_config.estimator_config.correction.enabled = false;
    online_config.estimator_config.selector = Default::default();
    online_config.estimator_config.fpm_learned.update_every = 2;
    let mut online = ForwardPassPerfModel::best_available(online_config.clone()).unwrap();
    online
        .tune_with_fpms(&[vec![metric(0, 2, 1000, 20.0)]])
        .unwrap();
    let runtime = dir.path().join("online-runtime");
    online.save_online_checkpoint(&runtime).unwrap();
    online_config.estimator_config.fpm_learned.checkpoint_path = Some(runtime.clone());
    let mut resumed = ForwardPassPerfModel::best_available(online_config).unwrap();
    online.flush_updates().unwrap();
    resumed.flush_updates().unwrap();
    close(
        estimate(&online, &[metric(0, 2, 1000, 0.0)]),
        estimate(&resumed, &[metric(0, 2, 1000, 0.0)]),
    );
    let mut invalid_combined = config;
    invalid_combined
        .estimator_config
        .fpm_learned
        .checkpoint_path = Some(runtime);
    assert!(
        ForwardPassPerfModel::best_available(invalid_combined)
            .unwrap_err()
            .to_string()
            .contains("requires a training checkpoint")
    );
    assert_eq!(
        bincode::serialize(&super::super::super::LearnedAdaptation::Frozen).unwrap(),
        [0, 0, 0, 0]
    );
    assert_eq!(
        bincode::serialize(&super::super::super::LearnedAdaptation::Online).unwrap(),
        [1, 0, 0, 0]
    );
}
