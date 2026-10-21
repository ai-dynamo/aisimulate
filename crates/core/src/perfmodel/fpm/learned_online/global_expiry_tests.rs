// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{EstimationMode, ForwardPassMetrics, ForwardPassPerfModel, ScheduledRequestMetrics};

fn aggregated_fixture(
    refresh: LearnedDerivativeRefresh,
) -> (
    OnlineLearner,
    LearnedForwardPassModel,
    ForwardPassPerfModelConfig,
) {
    let mut controls = controls(LearnedUpdateMode::LeafOnly, refresh);
    controls.update_every = 2;
    controls.recent_window = 16;
    controls.recent_max_age_observations = Some(3);
    let (mut learner, seed) = fixture(
        controls.clone(),
        &[&[1.0]],
        &[10.0],
        10.0,
        vec![leaf(0.0)],
        settings(1.0, 0.0),
    );
    learner
        .stores
        .insert(Kind::PurePrefill, learner.stores[&KIND].clone());
    let mut artifact = seed.artifact();
    artifact.worker_type = ForwardPassWorkerType::Aggregated;
    artifact
        .stores
        .insert(Kind::PurePrefill, artifact.stores[&KIND].clone());
    let seed = LearnedForwardPassModel::from_artifact(artifact).unwrap();
    let mut config = canonical();
    config.worker_type = ForwardPassWorkerType::Aggregated;
    config.estimation_mode = EstimationMode::FpmLearned;
    config.estimator_config.fpm_learned = controls;
    config.estimator_config.correction.enabled = false;
    learner.identity_json = canonical_learned_identity(&config).unwrap().to_string();
    (learner, seed, config)
}

fn public_model(
    refresh: LearnedDerivativeRefresh,
) -> (
    tempfile::TempDir,
    ForwardPassPerfModelConfig,
    ForwardPassPerfModel,
) {
    let (learner, seed, mut config) = aggregated_fixture(refresh);
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("seed");
    learner.save(&path, &seed, "{}").unwrap();
    config.estimator_config.fpm_learned.checkpoint_path = Some(path);
    let model = ForwardPassPerfModel::best_available(config.clone()).unwrap();
    (directory, config, model)
}

fn observation(prefill: bool, wall_ms: f64) -> ForwardPassMetrics {
    ForwardPassMetrics {
        wall_time: wall_ms / 1000.0,
        scheduled_requests: if prefill {
            ScheduledRequestMetrics {
                num_prefill_requests: 1,
                sum_prefill_tokens: 128,
                ..Default::default()
            }
        } else {
            ScheduledRequestMetrics {
                num_decode_requests: 1,
                sum_decode_kv_tokens: 128,
                ..Default::default()
            }
        },
        ..Default::default()
    }
}

fn admit(model: &mut ForwardPassPerfModel, prefill: bool, wall_ms: f64) {
    model
        .tune_with_fpms(&[vec![observation(prefill, wall_ms)]])
        .unwrap();
}

fn predict_prefill(model: &ForwardPassPerfModel) -> f64 {
    model
        .estimate_forward_pass_time_ms(&[observation(true, 0.0)])
        .unwrap()
        .unwrap()
}

#[test]
fn public_global_age_flush_expires_stale_stores_and_resumes_exactly() {
    for refresh in [
        LearnedDerivativeRefresh::Eager,
        LearnedDerivativeRefresh::Lazy,
    ] {
        // One row is only pending; two rows are only fitted; three leaves both.
        // All must expire after admissions in a different store.
        for prefill_count in 1..=3 {
            let (directory, mut config, mut model) = public_model(refresh);
            for label in [20.0, 30.0, 3000.0].into_iter().take(prefill_count) {
                admit(&mut model, true, label);
            }
            for label in [40.0, 50.0, 60.0, 70.0, 80.0, 90.0] {
                admit(&mut model, false, label);
            }
            // Queries and checkpoint saves do not expire stale evidence.
            close(
                predict_prefill(&model),
                if prefill_count == 1 { 10.0 } else { 20.0 },
            );
            let before = model.learned_online_diagnostics().unwrap();
            assert_eq!(
                before["stores"]["pure_prefill"]["pending_rows"],
                prefill_count % 2
            );
            let path = directory.path().join("pending");
            model.save_online_checkpoint(&path).unwrap();
            config.estimator_config.fpm_learned.checkpoint_path = Some(path);
            let mut resumed = ForwardPassPerfModel::best_available(config.clone()).unwrap();
            assert_eq!(
                model.export_learned_artifact().unwrap(),
                resumed.export_learned_artifact().unwrap()
            );
            for model in [&mut model, &mut resumed] {
                model.flush_updates().unwrap();
                // Every prefill label has expired; only the 10 ms anchor remains.
                close(predict_prefill(model), 10.0);
                let diagnostics = model.learned_online_diagnostics().unwrap();
                let store = &diagnostics["stores"]["pure_prefill"];
                assert_eq!(store["retained_row_ids"], serde_json::json!([]));
                assert_eq!(store["pending_rows"], 0);
                assert_eq!(store["accepted_observations"], prefill_count);
                assert_eq!(store["evicted_observations"], prefill_count);
                assert_eq!(
                    store["publications"],
                    if prefill_count == 1 { 0 } else { 2 }
                );
                assert_eq!(
                    diagnostics["stores"]["pure_decode"]["retained_row_ids"],
                    serde_json::json!([prefill_count + 4, prefill_count + 5, prefill_count + 6])
                );
                let version = diagnostics["model_version"].clone();
                model.flush_updates().unwrap();
                assert_eq!(
                    model.learned_online_diagnostics().unwrap()["model_version"],
                    version
                );
                // New rows get the same global IDs after checkpoint resumption.
                admit(model, true, 50.0);
                admit(model, true, 70.0);
                close(predict_prefill(model), 130.0 / 3.0);
                assert_eq!(
                    model.learned_online_diagnostics().unwrap()["stores"]["pure_prefill"]["retained_row_ids"],
                    serde_json::json!([prefill_count + 7, prefill_count + 8])
                );
            }
            assert_eq!(
                model.export_learned_artifact().unwrap(),
                resumed.export_learned_artifact().unwrap()
            );
            let path = directory.path().join("expired-and-refitted");
            resumed.save_online_checkpoint(&path).unwrap();
            config.estimator_config.fpm_learned.checkpoint_path = Some(path);
            let restored = ForwardPassPerfModel::best_available(config).unwrap();
            close(predict_prefill(&restored), 130.0 / 3.0);
        }
    }
}

#[test]
fn public_global_age_expiry_restores_history_when_history_weight_is_zero() {
    for refresh in [
        LearnedDerivativeRefresh::Eager,
        LearnedDerivativeRefresh::Lazy,
    ] {
        let (mut learner, seed, mut config) = aggregated_fixture(refresh);
        learner.config.history_weight = Some(0.0);
        config.estimator_config.fpm_learned.history_weight = Some(0.0);
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("seed");
        learner.save(&path, &seed, "{}").unwrap();
        config.estimator_config.fpm_learned.checkpoint_path = Some(path);
        let mut model = ForwardPassPerfModel::best_available(config).unwrap();

        for _ in 0..3 {
            admit(&mut model, true, 30.0);
        }
        for _ in 0..6 {
            admit(&mut model, false, 50.0);
        }
        // With recent evidence present, beta=0 gives the 10 ms anchor no mass.
        close(predict_prefill(&model), 30.0);
        model.flush_updates().unwrap();
        // IDs 1, 2, and 3 have all expired against global admission ID 9.
        // With no recent evidence left, the history-only objective is restored.
        close(predict_prefill(&model), 10.0);
        let diagnostics = model.learned_online_diagnostics().unwrap();
        let store = &diagnostics["stores"]["pure_prefill"];
        assert_eq!(store["retained_row_ids"], serde_json::json!([]));
        assert_eq!(store["recent_rows"], 0);
        assert_eq!(store["pending_rows"], 0);
        assert_eq!(store["accepted_observations"], 3);
        assert_eq!(store["evicted_observations"], 3);
        assert_eq!(store["history_row_weight"], 1.0);
    }
}

#[test]
fn public_global_age_counts_accepted_rows_and_preserves_inclusive_boundary() {
    for refresh in [
        LearnedDerivativeRefresh::Eager,
        LearnedDerivativeRefresh::Lazy,
    ] {
        let (_directory, _config, mut model) = public_model(refresh);
        admit(&mut model, true, 20.0); // ID 1.
        admit(&mut model, true, 40.0); // ID 2: prefill publishes.
        admit(&mut model, false, 50.0); // ID 3: oldest retained ID is 1.
        for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            admit(&mut model, false, invalid);
        }
        model
            .tune_with_fpms(&[vec![ForwardPassMetrics::default()]])
            .unwrap();
        assert!(model.tune_with_fpms(&[vec![]]).is_err());
        model.flush_updates().unwrap();
        let diagnostics = model.learned_online_diagnostics().unwrap();
        assert_eq!(diagnostics["rejected_observations"], 5);
        assert_eq!(
            diagnostics["stores"]["pure_prefill"]["retained_row_ids"],
            serde_json::json!([1, 2])
        );
        close(predict_prefill(&model), 70.0 / 3.0);
        admit(&mut model, false, 60.0); // ID 4: ID 1 now expires globally.
        model.flush_updates().unwrap();
        assert_eq!(
            model.learned_online_diagnostics().unwrap()["stores"]["pure_prefill"]["retained_row_ids"],
            serde_json::json!([2])
        );
        close(predict_prefill(&model), 25.0); // mean(anchor=10, retained label=40).
    }
}

#[test]
fn public_automatic_batch_excludes_pending_labels_expired_by_other_stores() {
    for refresh in [
        LearnedDerivativeRefresh::Eager,
        LearnedDerivativeRefresh::Lazy,
    ] {
        let (_directory, _config, mut model) = public_model(refresh);
        admit(&mut model, true, 3000.0); // ID 1, pending only.
        for label in [20.0, 30.0, 40.0] {
            admit(&mut model, false, label); // IDs 2, 3, 4.
        }
        admit(&mut model, true, 50.0); // ID 5 triggers the prefill batch.
        close(predict_prefill(&model), 30.0); // mean(anchor=10, unexpired label=50).
        let diagnostics = model.learned_online_diagnostics().unwrap();
        assert_eq!(
            diagnostics["stores"]["pure_prefill"]["retained_row_ids"],
            serde_json::json!([5])
        );
        assert_eq!(
            diagnostics["stores"]["pure_prefill"]["evicted_observations"],
            1
        );
    }
}

#[test]
fn global_age_removal_failure_preserves_fitted_rows_and_allows_retry() {
    for refresh in [
        LearnedDerivativeRefresh::Eager,
        LearnedDerivativeRefresh::Lazy,
    ] {
        let (mut learner, seed, _) = aggregated_fixture(refresh);
        let mut artifact = seed.artifact();
        artifact.target = LearnedTarget::LogMs;
        for store in artifact.stores.values_mut() {
            store.baseline = 0.0;
            store.trees[0].value[0] = 10.0;
        }
        let mut model = LearnedForwardPassModel::from_artifact(artifact).unwrap();
        for (&kind, state) in &mut learner.stores {
            fitting::bootstrap(state, model.store(kind).unwrap(), &learner.config).unwrap();
        }
        for _ in 0..2 {
            learner
                .observe(Kind::PurePrefill, vec![1.0], 1.0, &mut model)
                .unwrap();
        }
        for _ in 0..6 {
            learner.observe(KIND, vec![1.0], 1.0, &mut model).unwrap();
        }
        learner
            .stores
            .get_mut(&Kind::PurePrefill)
            .unwrap()
            .settings
            .learning_rate = 100.0;
        let before = state_bytes(&learner.stores[&Kind::PurePrefill]);
        let predictor = model.artifact();
        let version = learner.version;
        // Removing all recent labels leaves target 10 at learning rate 100:
        // raw prediction 1000 cannot pass through the exponential link.
        assert!(
            learner
                .flush(Kind::PurePrefill, &mut model)
                .unwrap_err()
                .to_string()
                .contains("prediction range can overflow")
        );
        assert_eq!(state_bytes(&learner.stores[&Kind::PurePrefill]), before);
        assert_eq!(model.artifact(), predictor);
        assert_eq!(learner.version, version);
        assert_eq!(learner.stores[&Kind::PurePrefill].counters.evicted, 0);
        learner
            .stores
            .get_mut(&Kind::PurePrefill)
            .unwrap()
            .settings
            .learning_rate = 1.0;
        assert!(learner.flush(Kind::PurePrefill, &mut model).unwrap());
        assert!(learner.stores[&Kind::PurePrefill].recent.is_empty());
        assert_eq!(learner.stores[&Kind::PurePrefill].counters.evicted, 2);
        checkpoint::validate_state(&learner, &model).unwrap();
    }
}
