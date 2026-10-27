// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::ForwardPassWorkerType;
use super::super::learned::{
    LEARNED_SCHEMA_NAME, LEARNED_SCHEMA_VERSION, LearnedArtifact, LearnedStore,
};
use super::*;

#[path = "frozen_prefix_tests.rs"]
mod frozen_prefix_tests;
#[path = "global_expiry_tests.rs"]
mod global_expiry_tests;
#[path = "grouped_history_tests.rs"]
mod grouped_history_tests;

const KIND: Kind = Kind::PureDecode;

fn leaf(value: f64) -> LearnedTree {
    LearnedTree {
        left: vec![-1],
        right: vec![-1],
        feature: vec![-1],
        threshold: vec![0.0],
        value: vec![value],
        missing_left: vec![true],
    }
}

fn split(feature: i32, missing_left: bool) -> LearnedTree {
    LearnedTree {
        left: vec![1, -1, -1],
        right: vec![2, -1, -1],
        feature: vec![feature, -1, -1],
        threshold: vec![0.5, 0.0, 0.0],
        value: vec![0.0; 3],
        missing_left: vec![missing_left, true, true],
    }
}

fn controls(mode: LearnedUpdateMode, refresh: LearnedDerivativeRefresh) -> FpmLearnedConfig {
    FpmLearnedConfig {
        adaptation: LearnedAdaptation::Online,
        update_mode: mode,
        derivative_refresh: refresh,
        update_every: 1,
        recent_window: 8,
        // Numerical oracles below exercise unit-weight fitting independently
        // of the public gym recipe.
        history_weight: None,
        history_weight_ramp_observations: 0,
        history_bin_grouping: false,
        split_sampling_rate: 1.0,
        split_rank_tolerance: 0.0,
        ..FpmLearnedConfig::default()
    }
}

fn canonical() -> ForwardPassPerfModelConfig {
    // Deserializing the public schema avoids any system database dependency.
    serde_json::from_value(serde_json::json!({
        "model": "unit-test-model", "system": "unit-test-system",
        "backend": "sglang", "worker_type": "decode"
    }))
    .unwrap()
}

fn fixture(
    config: FpmLearnedConfig,
    xs: &[&[f64]],
    ys: &[f64],
    baseline: f64,
    trees: Vec<LearnedTree>,
    settings: FitSettings,
) -> (OnlineLearner, LearnedForwardPassModel) {
    assert_eq!(xs.len(), ys.len());
    let width = xs[0].len();
    let thresholds = vec![vec![0.5]; width];
    let anchors = Arc::new(Anchors {
        ids: (0..ys.len() as u64).collect(),
        y: ys.to_vec(),
        x: xs.iter().flat_map(|x| x.iter().copied()).collect(),
        bins: xs.iter().flat_map(|x| bin_row(x, &thresholds)).collect(),
        width,
        groups: None,
    });
    let mut state = StoreState {
        anchors,
        history_grouping_ms: None,
        thresholds,
        offsets: (0..=width).map(|i| 3 * i).collect(),
        settings,
        trees: Vec::new(),
        recent: VecDeque::new(),
        pending: Vec::new(),
        frozen_anchor_prefix: Vec::new(),
        counters: Counters::default(),
    };
    let store = LearnedStore { baseline, trees };
    fitting::bootstrap(&mut state, &store, &config).unwrap();
    let features = ["num_decode_requests", "sum_decode_kv_tokens"];
    let model = LearnedForwardPassModel::from_artifact(LearnedArtifact {
        schema: LEARNED_SCHEMA_NAME.into(),
        schema_version: LEARNED_SCHEMA_VERSION,
        worker_type: ForwardPassWorkerType::Decode,
        target: LearnedTarget::Ms,
        features: features[..width].iter().map(|x| (*x).into()).collect(),
        stores: BTreeMap::from([(KIND, store)]),
        metadata: serde_json::json!({}),
    })
    .unwrap();
    let mut identity = serde_json::to_value(canonical()).unwrap();
    for key in [
        "estimator_config",
        "estimation_mode",
        "fallback_policy",
        "systems_paths",
        "database_mode",
        "transfer_policy",
        "strict_provenance",
    ] {
        identity.as_object_mut().unwrap().remove(key);
    }
    let learner = OnlineLearner {
        config,
        identity_json: serde_json::to_string(&identity).unwrap(),
        anchor_manifest_sha256: "fixture-anchor-hash".into(),
        stores: BTreeMap::from([(KIND, state)]),
        next_row_id: ys.len() as u64,
        version: 0,
        rejected_observations: 0,
        restored_corrections: None,
    };
    (learner, model)
}

fn settings(learning_rate: f64, l2_regularization: f64) -> FitSettings {
    FitSettings {
        learning_rate,
        l2_regularization,
        min_samples_leaf: 1,
        max_depth: None,
    }
}

fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
}

fn state_bytes(state: &StoreState) -> Vec<u8> {
    let mut state = state.clone();
    state.counters = Counters::default();
    bincode::serialize(&state).unwrap()
}

#[test]
fn eager_squared_loss_applies_l2_and_shrinkage_once() {
    let (mut learner, mut model) = fixture(
        controls(LearnedUpdateMode::LeafOnly, LearnedDerivativeRefresh::Eager),
        &[&[0.0], &[1.0]],
        &[2.0, 6.0],
        1.0,
        vec![leaf(0.0)],
        settings(0.5, 2.0),
    );
    learner.observe(KIND, vec![0.25], 10.0, &mut model).unwrap();
    // G = (1-2)+(1-6)+(1-10) = -15, H=3. The exported value
    // already includes eta: -.5*(-15)/(3+2) = 1.5, raw prediction=2.5.
    close(model.store(KIND).unwrap().trees[0].value[0], 1.5);
    close(model.store(KIND).unwrap().baseline, 1.0);
    let state = &learner.stores[&KIND];
    close(state.trees[0].stats[0].total.g, -15.0);
    assert_eq!(state.trees[0].stats[0].total.count, 3);
    assert_eq!(state.counters.eager_updates, 1);
}

#[test]
fn eager_later_tree_uses_already_updated_prefix() {
    let (mut learner, mut model) = fixture(
        controls(LearnedUpdateMode::LeafOnly, LearnedDerivativeRefresh::Eager),
        &[&[0.0], &[1.0]],
        &[2.0, 6.0],
        1.0,
        vec![leaf(0.0), leaf(0.0)],
        settings(0.5, 2.0),
    );
    learner.observe(KIND, vec![0.25], 10.0, &mut model).unwrap();
    // First tree gives 1.5. Second gradients use prefix 2.5:
    // G=(2.5-2)+(2.5-6)+(2.5-10)=-10.5, value=1.05.
    let store = model.store(KIND).unwrap();
    close(store.trees[0].value[0], 1.5);
    close(store.trees[1].value[0], 1.05);
    close(learner.stores[&KIND].trees[1].stats[0].total.g, -10.5);
}

#[test]
fn fifo_evicts_oldest_online_rows_and_preserves_all_anchors() {
    let mut config = controls(LearnedUpdateMode::LeafOnly, LearnedDerivativeRefresh::Eager);
    config.recent_window = 2;
    let (mut learner, mut model) = fixture(
        config,
        &[&[0.0]],
        &[1.0],
        0.0,
        vec![leaf(0.0)],
        settings(1.0, 0.0),
    );
    for y in [3.0, 5.0, 7.0] {
        learner.observe(KIND, vec![1.0], y, &mut model).unwrap();
    }
    let state = &learner.stores[&KIND];
    assert_eq!(state.anchors.ids, vec![0]);
    assert_eq!(state.anchors.y, vec![1.0]);
    assert_eq!(
        state.recent.iter().map(|r| r.id).collect::<Vec<_>>(),
        vec![2, 3]
    );
    assert_eq!(state.counters.evicted, 1);
    close(model.store(KIND).unwrap().trees[0].value[0], 13.0 / 3.0);
    assert_eq!(state.trees[0].stats[0].total.count, 3);
}

#[test]
fn oversized_batch_retains_only_newest_rows() {
    let mut config = controls(LearnedUpdateMode::LeafOnly, LearnedDerivativeRefresh::Lazy);
    config.recent_window = 2;
    config.update_every = 4;
    let (mut learner, mut model) = fixture(
        config,
        &[&[0.0]],
        &[1.0],
        0.0,
        vec![leaf(0.0)],
        settings(1.0, 0.0),
    );
    for y in [3.0, 5.0, 7.0, 9.0] {
        learner.observe(KIND, vec![1.0], y, &mut model).unwrap();
    }
    let state = &learner.stores[&KIND];
    assert_eq!(
        state.recent.iter().map(|r| r.y).collect::<Vec<_>>(),
        vec![7.0, 9.0]
    );
    assert_eq!(state.counters.evicted, 2);
    close(model.store(KIND).unwrap().trees[0].value[0], 17.0 / 3.0);
}

fn feature_change_fixture(
    refresh: LearnedDerivativeRefresh,
) -> (OnlineLearner, LearnedForwardPassModel) {
    fixture(
        controls(LearnedUpdateMode::Structural, refresh),
        &[&[0.0, 0.0], &[0.0, 1.0], &[1.0, 0.0], &[1.0, 1.0]],
        &[1.0, 9.0, 1.0, 9.0],
        0.0,
        vec![split(0, true)],
        settings(1.0, 0.0),
    )
}

#[test]
fn structural_rebuild_changes_feature_within_original_capacity() {
    let (mut learner, mut model) = feature_change_fixture(LearnedDerivativeRefresh::Eager);
    learner
        .observe(KIND, vec![0.0, 0.0], 1.0, &mut model)
        .unwrap();
    let tree = &model.store(KIND).unwrap().trees[0];
    assert_eq!(tree.feature[0], 1);
    assert_eq!(tree.value.len(), 3);
    close(tree.value[tree.left[0] as usize], 1.0);
    close(tree.value[tree.right[0] as usize], 9.0);
    assert_eq!(learner.stores[&KIND].trees[0].regions[0], vec![0, 1, 2]);
    assert!(learner.stores[&KIND].counters.rebuilt_nodes > 0);
}

#[test]
fn invalid_partition_collapses_then_reuses_reserved_slots() {
    let mut config = controls(
        LearnedUpdateMode::Structural,
        LearnedDerivativeRefresh::Eager,
    );
    config.recent_window = 1;
    let (mut learner, mut model) = fixture(
        config,
        &[&[0.0]],
        &[1.0],
        0.0,
        vec![split(0, true)],
        settings(1.0, 0.0),
    );
    learner.observe(KIND, vec![1.0], 9.0, &mut model).unwrap();
    assert_eq!(model.store(KIND).unwrap().trees[0].left[0], 1);
    learner.observe(KIND, vec![0.0], 3.0, &mut model).unwrap();
    let collapsed = &model.store(KIND).unwrap().trees[0];
    assert_eq!(collapsed.left[0], -1);
    assert_eq!(collapsed.value.len(), 3);
    assert_eq!(learner.stores[&KIND].trees[0].regions[0], vec![0, 1, 2]);
    close(collapsed.value[0], 2.0);
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("collapsed");
    learner.save(&path, &model, "{}").unwrap();
    let mut resumed_model = checkpoint_model(&path).unwrap();
    let mut resumed =
        OnlineLearner::load(&path, &mut resumed_model, &learner.config, &canonical()).unwrap();
    learner.observe(KIND, vec![1.0], 9.0, &mut model).unwrap();
    resumed
        .observe(KIND, vec![1.0], 9.0, &mut resumed_model)
        .unwrap();
    assert_eq!(model.artifact(), resumed_model.artifact());
    assert_eq!(
        state_bytes(&learner.stores[&KIND]),
        state_bytes(&resumed.stores[&KIND])
    );
    let regrown = &model.store(KIND).unwrap().trees[0];
    assert_eq!(regrown.left[0], 1);
    assert_eq!(regrown.right[0], 2);
    assert_eq!(regrown.value.len(), 3);
    close(regrown.value[1], 1.0);
    close(regrown.value[2], 9.0);
}

#[test]
fn missing_direction_and_threshold_equality_match_bins_and_prediction() {
    assert_eq!(
        bin_row(&[0.5, f64::NAN], &[vec![0.5], vec![0.5]]),
        vec![0, 2]
    );
    assert_eq!(bin_row(&[0.5000000000000001], &[vec![0.5]]), vec![1]);
    for missing_left in [false, true] {
        let (mut learner, mut model) = fixture(
            controls(LearnedUpdateMode::LeafOnly, LearnedDerivativeRefresh::Eager),
            &[&[0.0], &[1.0], &[f64::NAN]],
            &[1.0, 9.0, 5.0],
            0.0,
            vec![split(0, missing_left)],
            settings(1.0, 0.0),
        );
        learner.observe(KIND, vec![0.5], 3.0, &mut model).unwrap();
        let tree = &model.store(KIND).unwrap().trees[0];
        assert_eq!(leaf_index(tree, &[0.5]), 1);
        assert_eq!(
            leaf_index(tree, &[f64::NAN]),
            if missing_left { 1 } else { 2 }
        );
        close(tree.value[1], if missing_left { 3.0 } else { 2.0 });
        close(tree.value[2], if missing_left { 9.0 } else { 7.0 });
    }
}

#[test]
fn lazy_rebuild_fallback_matches_direct_eager_state() {
    let (mut lazy, mut lazy_model) = feature_change_fixture(LearnedDerivativeRefresh::Lazy);
    let mut eager = lazy.clone();
    let mut eager_model = lazy_model.clone();
    // Keep the same persistent-gradient format, but force immediate eager fit.
    eager.config.eager_refresh_interval = 1;
    lazy.observe(KIND, vec![0.0, 0.0], 1.0, &mut lazy_model)
        .unwrap();
    eager
        .observe(KIND, vec![0.0, 0.0], 1.0, &mut eager_model)
        .unwrap();
    assert_eq!(lazy_model.artifact(), eager_model.artifact());
    assert_eq!(
        state_bytes(&lazy.stores[&KIND]),
        state_bytes(&eager.stores[&KIND])
    );
    assert_eq!(lazy.stores[&KIND].counters.lazy_fallbacks, 1);
    assert_eq!(lazy.stores[&KIND].counters.eager_updates, 1);
}

#[test]
fn lazy_historical_aggregates_stay_stale_until_the_32nd_batch() {
    let config = controls(LearnedUpdateMode::LeafOnly, LearnedDerivativeRefresh::Lazy);
    assert_eq!(config.eager_refresh_interval, 32);
    let (mut learner, mut model) = fixture(
        config,
        &[&[0.0]],
        &[1.0],
        0.0,
        vec![leaf(0.0), leaf(0.0)],
        settings(0.5, 0.0),
    );
    for _ in 0..31 {
        learner.observe(KIND, vec![1.0], 3.0, &mut model).unwrap();
    }
    let state = &learner.stores[&KIND];
    assert_eq!(state.counters.lazy_updates, 31);
    assert_eq!(state.counters.eager_updates, 0);
    let recent_gradient: f64 = state.recent.iter().map(|row| row.gradients[1]).sum();
    // Anchor derivatives remain stale in the aggregate; no per-anchor cache
    // is needed because only recent rows can be removed incrementally.
    close(state.trees[1].stats[0].total.g - recent_gradient, -1.0);
    learner.observe(KIND, vec![1.0], 3.0, &mut model).unwrap();
    let state = &learner.stores[&KIND];
    assert_eq!(state.counters.eager_updates, 1);
    assert_eq!(state.counters.batches_since_eager, 0);
    // Retained labels sum to 25 (one anchor + eight recent); first leaf=25/18.
    let recent_gradient: f64 = state.recent.iter().map(|row| row.gradients[1]).sum();
    close(
        state.trees[1].stats[0].total.g - recent_gradient,
        25.0 / 18.0 - 1.0,
    );
}

#[test]
fn failed_lazy_batch_restores_float_bits_and_keeps_pending_for_retry() {
    let mut config = controls(LearnedUpdateMode::LeafOnly, LearnedDerivativeRefresh::Lazy);
    config.update_every = 2;
    let (mut learner, mut model) = fixture(
        config,
        &[&[0.0]],
        &[1.0],
        0.0,
        vec![leaf(0.0)],
        settings(f64::MAX, 0.0),
    );
    learner.observe(KIND, vec![1.0], 3.0, &mut model).unwrap();
    let before_state = state_bytes(&learner.stores[&KIND]);
    let before_model = model.artifact();
    let before_version = learner.version;
    // eta*G overflows after the incremental aggregate has already changed.
    assert!(learner.flush(KIND, &mut model).is_err());
    assert_eq!(state_bytes(&learner.stores[&KIND]), before_state);
    assert_eq!(model.artifact(), before_model);
    assert_eq!(learner.version, before_version);
    assert_eq!(learner.stores[&KIND].pending.len(), 1);
    assert_eq!(learner.stores[&KIND].counters.failures, 1);
    learner
        .stores
        .get_mut(&KIND)
        .unwrap()
        .settings
        .learning_rate = 0.5;
    assert!(learner.flush(KIND, &mut model).unwrap());
    close(model.store(KIND).unwrap().trees[0].value[0], 1.0);
}

#[test]
fn clones_share_immutable_anchors_but_fit_independently() {
    let (original, model) = fixture(
        controls(LearnedUpdateMode::LeafOnly, LearnedDerivativeRefresh::Lazy),
        &[&[0.0]],
        &[1.0],
        0.0,
        vec![leaf(0.0)],
        settings(1.0, 0.0),
    );
    let original_bytes = bincode::serialize(&original).unwrap();
    let mut clone = original.clone();
    let mut clone_model = model.clone();
    assert!(Arc::ptr_eq(
        &original.stores[&KIND].anchors,
        &clone.stores[&KIND].anchors
    ));
    clone
        .observe(KIND, vec![1.0], 3.0, &mut clone_model)
        .unwrap();
    assert_eq!(bincode::serialize(&original).unwrap(), original_bytes);
    close(model.store(KIND).unwrap().trees[0].value[0], 0.0);
    close(clone_model.store(KIND).unwrap().trees[0].value[0], 2.0);
}

#[test]
fn runtime_checkpoint_preserves_pending_stale_derivatives_and_continuation() {
    let mut config = controls(LearnedUpdateMode::LeafOnly, LearnedDerivativeRefresh::Lazy);
    config.update_every = 2;
    let (mut learner, mut model) = fixture(
        config,
        &[&[0.0]],
        &[1.0],
        0.0,
        vec![leaf(0.0), leaf(0.0)],
        settings(0.5, 0.0),
    );
    for y in [3.0, 4.0, 7.0] {
        learner.observe(KIND, vec![1.0], y, &mut model).unwrap();
    }
    assert_eq!(learner.stores[&KIND].pending.len(), 1);
    let state = &learner.stores[&KIND];
    let recent_gradient: f64 = state.recent.iter().map(|row| row.gradients[1]).sum();
    close(state.trees[1].stats[0].total.g - recent_gradient, -1.0);
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("saved");
    let corrections = "{\"ratio\":1.125,\"observations\":3}";
    learner.save(&path, &model, corrections).unwrap();
    let mut resumed_model = checkpoint_model(&path).unwrap();
    let mut resumed =
        OnlineLearner::load(&path, &mut resumed_model, &learner.config, &canonical()).unwrap();
    assert_eq!(resumed.restored_correction_json(), Some(corrections));
    assert_eq!(
        state_bytes(&resumed.stores[&KIND]),
        state_bytes(&learner.stores[&KIND])
    );
    for y in [9.0, 11.0, 13.0] {
        learner.observe(KIND, vec![1.0], y, &mut model).unwrap();
        resumed
            .observe(KIND, vec![1.0], y, &mut resumed_model)
            .unwrap();
    }
    assert_eq!(resumed_model.artifact(), model.artifact());
    assert_eq!(
        state_bytes(&resumed.stores[&KIND]),
        state_bytes(&learner.stores[&KIND])
    );
    assert_eq!(resumed.next_row_id, learner.next_row_id);
    assert_eq!(resumed.version, learner.version);
}

#[test]
fn runtime_checkpoint_rejects_wrong_identity_changed_controls_and_checksum() {
    let (learner, model) = fixture(
        controls(LearnedUpdateMode::LeafOnly, LearnedDerivativeRefresh::Lazy),
        &[&[0.0]],
        &[1.0],
        0.0,
        vec![leaf(0.0)],
        settings(1.0, 0.0),
    );
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("saved");
    learner.save(&path, &model, "{}").unwrap();
    let mut wrong_identity = canonical();
    wrong_identity.tp = 2;
    let error = OnlineLearner::load(&path, &mut model.clone(), &learner.config, &wrong_identity)
        .unwrap_err();
    assert!(error.to_string().contains("identity"));
    let mut changed = learner.config.clone();
    changed.seed += 1;
    let error = OnlineLearner::load(&path, &mut model.clone(), &changed, &canonical()).unwrap_err();
    assert!(error.to_string().contains("controls differ"));
    let binary = path.join("state.bin");
    let mut bytes = std::fs::read(&binary).unwrap();
    bytes[0] ^= 1;
    std::fs::write(binary, bytes).unwrap();
    let error =
        OnlineLearner::load(&path, &mut model.clone(), &learner.config, &canonical()).unwrap_err();
    assert!(error.to_string().contains("checksum mismatch"));
}

#[test]
fn canonical_learned_checkpoint_rejects_native_only_controls() {
    let controls = controls(LearnedUpdateMode::LeafOnly, LearnedDerivativeRefresh::Eager);
    let (learner, seed_model) = fixture(
        controls.clone(),
        &[&[1.0]],
        &[10.0],
        10.0,
        vec![leaf(0.0)],
        settings(1.0, 0.0),
    );
    let directory = tempfile::tempdir().unwrap();
    let seed_path = directory.path().join("seed");
    learner.save(&seed_path, &seed_model, "{}").unwrap();
    let mut config = canonical();
    config.estimation_mode = crate::EstimationMode::FpmLearned;
    config.estimator_config.fpm_learned = controls;
    config.estimator_config.fpm_learned.checkpoint_path = Some(seed_path);
    config.estimator_config.correction.enabled = false;

    for adaptation in [LearnedAdaptation::Frozen, LearnedAdaptation::Online] {
        config.estimator_config.fpm_learned.adaptation = adaptation;
        crate::ForwardPassPerfModel::best_available(config.clone()).unwrap();

        // Neither learned mode prices a vision tower. Reject the layout before
        // checkpoint identity validation or the learned-mode early return.
        let mut vision = config.clone();
        vision.encoder_parallel = Some(crate::EncoderParallel::Tp);
        let error = crate::ForwardPassPerfModel::best_available(vision).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("encoder_parallel requires op_level timing"),
            "{adaptation:?}: {error}"
        );

        // External operation profiles must not be silently ignored by a forest.
        let mut external = config.clone();
        external.estimator_config.op_level.execution_profile = Some(
            crate::perfmodel::execution_profile::ExecutionProfileSelection {
                path: directory.path().join("unused-execution-profile"),
                sha256: "0".repeat(64),
            },
        );
        let error = crate::ForwardPassPerfModel::best_available(external).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("requires explicit op_level/SILICON"),
            "{adaptation:?}: {error}"
        );
    }
}

#[test]
fn public_online_publication_and_checkpoint_resume_pending() {
    use crate::{
        EstimationMode, ForwardPassMetrics, ForwardPassPerfModel, ScheduledRequestMetrics,
    };

    let mut online_config = controls(LearnedUpdateMode::LeafOnly, LearnedDerivativeRefresh::Eager);
    online_config.update_every = 2;
    let (learner, seed_model) = fixture(
        online_config.clone(),
        &[&[1.0]],
        &[10.0],
        10.0,
        vec![leaf(0.0)],
        settings(1.0, 0.0),
    );
    let mut config = canonical();
    config.estimation_mode = EstimationMode::FpmLearned;
    config.estimator_config.fpm_learned = online_config;
    config.estimator_config.correction.enabled = false;
    let correction_state = serde_json::json!({});
    let directory = tempfile::tempdir().unwrap();
    let seed_path = directory.path().join("seed");
    learner
        .save(&seed_path, &seed_model, &correction_state.to_string())
        .unwrap();
    config.estimator_config.fpm_learned.checkpoint_path = Some(seed_path);
    let mut model = ForwardPassPerfModel::best_available(config.clone()).unwrap();
    let exported: serde_json::Value =
        serde_json::from_str(&model.export_learned_artifact().unwrap()).unwrap();
    assert_eq!(
        exported["metadata"]["identity"],
        canonical_learned_identity(&config).unwrap()
    );
    let frozen_artifact = directory.path().join("verified-frozen.json");
    std::fs::write(&frozen_artifact, serde_json::to_vec(&exported).unwrap()).unwrap();
    let mut wrong_identity = config.clone();
    wrong_identity.system = "different-system".into();
    wrong_identity.estimator_config.fpm_learned.adaptation = LearnedAdaptation::Frozen;
    wrong_identity.estimator_config.fpm_learned.checkpoint_path = None;
    wrong_identity.estimator_config.fpm_learned.artifact_path = Some(frozen_artifact);
    wrong_identity.estimator_config.fpm_learned.source_sha256 = None;
    assert!(
        ForwardPassPerfModel::best_available(wrong_identity)
            .unwrap_err()
            .to_string()
            .contains("metadata.identity")
    );

    let observation = |wall_ms: f64| ForwardPassMetrics {
        wall_time: wall_ms / 1000.0,
        scheduled_requests: ScheduledRequestMetrics {
            num_decode_requests: 1,
            sum_decode_kv_tokens: 100,
            ..Default::default()
        },
        ..Default::default()
    };
    let estimate = |model: &ForwardPassPerfModel| {
        model
            .estimate_forward_pass_time_ms(&[observation(0.0)])
            .unwrap()
            .unwrap()
    };
    // Invalid telemetry is rejected before OnlineLearner::observe; idle work
    // is skipped without becoming a rejected or admitted observation.
    for wall_ms in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        model.tune_with_fpms(&[vec![observation(wall_ms)]]).unwrap();
    }
    model
        .tune_with_fpms(&[vec![ForwardPassMetrics::default()]])
        .unwrap();
    assert!(model.tune_with_fpms(&[vec![]]).is_err());
    let ingestion = model.learned_online_diagnostics().unwrap();
    assert_eq!(ingestion["rejected_observations"], 5);
    assert_eq!(
        ingestion["stores"]["pure_decode"]["accepted_observations"],
        0
    );
    close(estimate(&model), 10.0);
    model.tune_with_fpms(&[vec![observation(20.0)]]).unwrap();
    close(estimate(&model), 10.0); // A partial batch has not published.
    assert_eq!(model.diagnostics().retained_observations, 1);
    model.tune_with_fpms(&[vec![observation(30.0)]]).unwrap();
    close(estimate(&model), 20.0); // Published mean(10,20,30), old correction discarded.
    assert_eq!(model.diagnostics().retained_observations, 2);
    model.tune_with_fpms(&[vec![observation(40.0)]]).unwrap();
    close(estimate(&model), 20.0);
    let learner_diagnostics = |model: &ForwardPassPerfModel| {
        model.learned_online_diagnostics().map(|mut value| {
            // Optional process-wide profiling includes other model instances
            // and parallel tests; it is deliberately outside learner state.
            value.as_object_mut().unwrap().remove("profiling");
            value
        })
    };
    let diagnostics = learner_diagnostics(&model);
    let corrections = model.diagnostics();
    for _ in 0..5 {
        close(estimate(&model), 20.0);
    }
    assert_eq!(learner_diagnostics(&model), diagnostics);
    assert_eq!(model.diagnostics(), corrections);

    let resumed_path = directory.path().join("resumed");
    model.save_online_checkpoint(&resumed_path).unwrap();
    config.estimator_config.fpm_learned.checkpoint_path = Some(resumed_path.clone());
    let mut resumed = ForwardPassPerfModel::best_available(config.clone()).unwrap();
    close(estimate(&resumed), 20.0);
    assert_eq!(learner_diagnostics(&resumed), learner_diagnostics(&model));
    model.flush_updates().unwrap();
    resumed.flush_updates().unwrap();
    close(estimate(&model), 25.0); // mean(10,20,30,40), no correction against old base.
    close(estimate(&resumed), 25.0);
    assert_eq!(model.diagnostics().retained_observations, 3);
    assert_eq!(
        resumed.export_learned_artifact().unwrap(),
        model.export_learned_artifact().unwrap()
    );

    // Frozen use reads the saved forest only, even when online controls differ.
    config.estimator_config.fpm_learned.adaptation = LearnedAdaptation::Frozen;
    config.estimator_config.fpm_learned.seed = 987;
    let frozen = ForwardPassPerfModel::best_available(config).unwrap();
    close(estimate(&frozen), 20.0);
    assert_eq!(
        frozen.learned_online_diagnostics().unwrap()["adaptation"],
        "frozen"
    );
}

#[test]
fn runtime_checkpoint_rejects_checksummed_but_inconsistent_state() {
    type Mutation = (&'static str, fn(&mut OnlineLearner));
    let (mut learner, mut model) = feature_change_fixture(LearnedDerivativeRefresh::Lazy);
    learner
        .observe(KIND, vec![0.0, 0.0], 1.0, &mut model)
        .unwrap();
    let cases: &[Mutation] = &[
        ("missing cached gradient", |l| {
            l.stores.get_mut(&KIND).unwrap().recent[0].gradients.clear()
        }),
        ("wrong row width", |l| {
            l.stores.get_mut(&KIND).unwrap().recent[0].x.clear()
        }),
        ("wrong row bin", |l| {
            l.stores.get_mut(&KIND).unwrap().recent[0].bins[0] = 100
        }),
        ("duplicate row ID", |l| {
            l.stores.get_mut(&KIND).unwrap().recent[0].id = 0
        }),
        ("invalid next ID", |l| l.next_row_id = 0),
        ("wrong target transform", |l| {
            l.stores.get_mut(&KIND).unwrap().recent[0].wall_ms = 2.0
        }),
        ("short offsets", |l| {
            l.stores.get_mut(&KIND).unwrap().offsets.pop();
        }),
        ("nonfinite threshold", |l| {
            l.stores.get_mut(&KIND).unwrap().thresholds[0][0] = f64::NAN
        }),
        ("invalid ridge", |l| {
            l.stores.get_mut(&KIND).unwrap().settings.l2_regularization = -1.0
        }),
        ("short histogram", |l| {
            l.stores.get_mut(&KIND).unwrap().trees[0].stats[0]
                .histogram
                .pop();
        }),
        ("wrong root count", |l| {
            l.stores.get_mut(&KIND).unwrap().trees[0].stats[0]
                .total
                .count += 1
        }),
        ("wrong histogram count", |l| {
            l.stores.get_mut(&KIND).unwrap().trees[0].stats[0].histogram[0].count += 1
        }),
        ("invalid candidate", |l| {
            l.stores.get_mut(&KIND).unwrap().trees[0].candidates[0][0].feature = 100
        }),
        ("missing incumbent", |l| {
            l.stores.get_mut(&KIND).unwrap().trees[0].candidates[0].clear()
        }),
        ("lost capacity", |l| {
            l.stores.get_mut(&KIND).unwrap().trees[0].regions[0].pop();
        }),
        ("crossed region", |l| {
            l.stores.get_mut(&KIND).unwrap().trees[0].regions[1].push(2)
        }),
        ("overflowing counts", |l| {
            l.stores.get_mut(&KIND).unwrap().counters.evicted = u64::MAX
        }),
        ("wrong version", |l| l.version += 1),
    ];
    let directory = tempfile::tempdir().unwrap();
    for (index, (name, mutate)) in cases.iter().enumerate() {
        let mut corrupt = learner.clone();
        mutate(&mut corrupt);
        let path = directory.path().join(index.to_string());
        // save recomputes real hashes: rejection must be semantic, not checksum based.
        corrupt.save(&path, &model, "{}").unwrap();
        assert!(
            OnlineLearner::load(&path, &mut model.clone(), &learner.config, &canonical()).is_err(),
            "accepted {name}"
        );
    }
}

#[test]
fn runtime_binary_is_bound_to_predictor_and_anchor_identity() {
    use sha2::{Digest, Sha256};
    let (learner, model) = fixture(
        controls(LearnedUpdateMode::LeafOnly, LearnedDerivativeRefresh::Lazy),
        &[&[0.0]],
        &[1.0],
        0.0,
        vec![leaf(0.0)],
        settings(1.0, 0.0),
    );
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("saved");
    learner.save(&path, &model, "{}").unwrap();
    let manifest_path = path.join("manifest.json");
    let original: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    let mut manifest = original.clone();
    manifest["anchor_manifest_sha256"] = "different-anchor-set".into();
    std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let error =
        OnlineLearner::load(&path, &mut model.clone(), &learner.config, &canonical()).unwrap_err();
    assert!(error.to_string().contains("anchor identity"));

    let mut artifact = model.artifact();
    artifact.stores.get_mut(&KIND).unwrap().trees[0].value[0] = 0.125;
    let changed_model = LearnedForwardPassModel::from_artifact(artifact).unwrap();
    let bytes = changed_model.to_json().unwrap().into_bytes();
    let mut manifest = original;
    manifest["artifact_sha256"] = format!("{:x}", Sha256::digest(&bytes)).into();
    std::fs::write(path.join("model.json"), bytes).unwrap();
    std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let error = OnlineLearner::load(
        &path,
        &mut changed_model.clone(),
        &learner.config,
        &canonical(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("predictor checksum"));
}

#[test]
fn eager_internal_aggregate_overflow_is_rejected_before_publication() {
    let mut config = controls(LearnedUpdateMode::LeafOnly, LearnedDerivativeRefresh::Eager);
    config.update_every = 3;
    let (mut learner, mut model) = fixture(
        config,
        &[&[0.0]],
        &[1.0],
        0.0,
        vec![split(0, true)],
        settings(0.25, 0.0),
    );
    learner.observe(KIND, vec![0.0], 1e308, &mut model).unwrap();
    learner.observe(KIND, vec![1.0], 1e308, &mut model).unwrap();
    let before = state_bytes(&learner.stores[&KIND]);
    let predictor = model.artifact();
    // Each leaf's G is finite, but the parent G=-2e308 is not.
    assert!(learner.flush(KIND, &mut model).is_err());
    assert_eq!(state_bytes(&learner.stores[&KIND]), before);
    assert_eq!(model.artifact(), predictor);
    assert_eq!(learner.version, 0);
}

#[test]
fn structural_stumps_match_independent_regularized_squared_error_oracle() {
    // The reference scans raw rows and evaluates the minimized loss directly.
    // It does not use histograms, cached derivatives, candidate gains, or fitting helpers.
    fn objective(
        rows: &[(Vec<f64>, f64)],
        members: &[usize],
        baseline: f64,
        ridge: f64,
    ) -> (f64, f64) {
        let adjustment = members.iter().map(|&i| rows[i].1 - baseline).sum::<f64>()
            / (members.len() as f64 + ridge);
        let loss = members
            .iter()
            .map(|&i| {
                let residual = baseline + adjustment - rows[i].1;
                0.5 * residual * residual
            })
            .sum::<f64>()
            + 0.5 * ridge * adjustment * adjustment;
        (loss, adjustment)
    }
    let mut random = 0x78d35a90462f1cbu64;
    let mut draw = || {
        random = random
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        random >> 32
    };
    for case in 0..48 {
        let mut rows = Vec::new();
        for i in 0..10 {
            let x0 = match draw() % 3 {
                0 => 0.0,
                1 => 1.0,
                _ => f64::NAN,
            };
            let x1 = match i {
                0 => 0.0,
                1 => 1.0,
                _ => match draw() % 3 {
                    0 => 0.0,
                    1 => 1.0,
                    _ => f64::NAN,
                },
            };
            let high = if x1.is_nan() { case % 2 == 0 } else { x1 > 0.5 };
            let y = if high { 9.0 } else { 1.0 } + (draw() % 3) as f64 * 0.1;
            rows.push((vec![x0, x1], y));
        }
        let ridge = [0.0, 0.5, 2.0][case % 3];
        let rate = 0.4;
        let baseline = 5.0;
        let xs: Vec<_> = rows[..9].iter().map(|row| row.0.as_slice()).collect();
        let ys: Vec<_> = rows[..9].iter().map(|row| row.1).collect();
        let (mut learner, mut model) = fixture(
            controls(
                LearnedUpdateMode::Structural,
                LearnedDerivativeRefresh::Eager,
            ),
            &xs,
            &ys,
            baseline,
            vec![split(0, true)],
            settings(rate, ridge),
        );
        learner
            .observe(KIND, rows[9].0.clone(), rows[9].1, &mut model)
            .unwrap();
        let tree = &model.store(KIND).unwrap().trees[0];
        assert!(tree.left[0] >= 0, "high-signal case {case} collapsed");
        let mut optimal_loss = f64::INFINITY;
        for feature in 0..2 {
            for (threshold, missing_left) in [(0.5, false), (0.5, true), (f64::MAX, false)] {
                let (left, right): (Vec<_>, Vec<_>) = (0..rows.len()).partition(|&i| {
                    let x = rows[i].0[feature];
                    if x.is_nan() {
                        missing_left
                    } else {
                        x <= threshold
                    }
                });
                if left.is_empty() || right.is_empty() {
                    continue;
                }
                let candidate_loss = objective(&rows, &left, baseline, ridge).0
                    + objective(&rows, &right, baseline, ridge).0;
                optimal_loss = optimal_loss.min(candidate_loss);
            }
        }
        let (left, right): (Vec<_>, Vec<_>) = (0..rows.len()).partition(|&i| {
            let x = rows[i].0[tree.feature[0] as usize];
            if x.is_nan() {
                tree.missing_left[0]
            } else {
                x <= tree.threshold[0]
            }
        });
        let (left_loss, left_adjustment) = objective(&rows, &left, baseline, ridge);
        let (right_loss, right_adjustment) = objective(&rows, &right, baseline, ridge);
        close(left_loss + right_loss, optimal_loss);
        close(tree.value[tree.left[0] as usize], rate * left_adjustment);
        close(tree.value[tree.right[0] as usize], rate * right_adjustment);
    }
}

#[test]
fn seed_fit_compatibility_checks_shrinkage_ridge_and_float32_rounding() {
    let mut first = split(0, true);
    first.value = vec![0.0, -1.0, 1.0];
    let mut second = split(0, true);
    second.value = vec![0.0, -0.75, 0.75];
    let (learner, model) = fixture(
        controls(LearnedUpdateMode::LeafOnly, LearnedDerivativeRefresh::Eager),
        &[&[0.0], &[1.0]],
        &[1.0, 9.0],
        5.0,
        vec![first, second],
        settings(0.5, 1.0),
    );
    let state = &learner.stores[&KIND];
    let store = model.store(KIND).unwrap();
    checkpoint::validate_seed_fit(state, store).unwrap();
    let mut wrong = state.clone();
    wrong.settings.learning_rate = 0.25;
    assert!(checkpoint::validate_seed_fit(&wrong, store).is_err());
    wrong = state.clone();
    wrong.settings.l2_regularization = 2.0;
    assert!(checkpoint::validate_seed_fit(&wrong, store).is_err());

    // sklearn rounds each derivative to f32 before histogram reductions.
    let mut rounded = split(0, true);
    rounded.value = vec![
        0.0,
        -0.25 * ((5.0_f64 - 1.1) as f32 as f64),
        -0.25 * ((5.0_f64 - 8.9) as f32 as f64),
    ];
    let (learner, model) = fixture(
        controls(LearnedUpdateMode::LeafOnly, LearnedDerivativeRefresh::Eager),
        &[&[0.0], &[1.0]],
        &[1.1, 8.9],
        5.0,
        vec![rounded],
        settings(0.5, 1.0),
    );
    checkpoint::validate_seed_fit(&learner.stores[&KIND], model.store(KIND).unwrap()).unwrap();
}

#[test]
fn rank_tolerance_can_keep_a_valid_nonoptimal_incumbent() {
    let (mut learner, mut model) = feature_change_fixture(LearnedDerivativeRefresh::Eager);
    learner.config.split_rank_tolerance = 1.0;
    learner
        .observe(KIND, vec![0.0, 0.0], 1.0, &mut model)
        .unwrap();
    let tree = &model.store(KIND).unwrap().trees[0];
    assert_eq!(tree.feature[0], 0);
    assert_eq!(learner.stores[&KIND].counters.rebuilt_nodes, 0);
    close(tree.value[1], 11.0 / 3.0);
    close(tree.value[2], 5.0);
}

#[test]
fn sampled_candidates_keep_incumbent_and_are_independent_of_rank_tolerance() {
    let mut config = controls(
        LearnedUpdateMode::Structural,
        LearnedDerivativeRefresh::Lazy,
    );
    config.split_sampling_rate = 0.25;
    config.seed = 31;
    let (mut learner, model) = fixture(
        config.clone(),
        &[&[0.0], &[1.0]],
        &[1.0, 9.0],
        5.0,
        vec![split(0, true)],
        settings(0.5, 1.0),
    );
    let state = learner.stores.get_mut(&KIND).unwrap();
    state.thresholds = vec![(1..10).map(|i| i as f64 / 10.0).collect()];
    state.offsets = vec![0, 11];
    state.anchors = Arc::new(Anchors {
        ids: state.anchors.ids.clone(),
        y: state.anchors.y.clone(),
        x: state.anchors.x.clone(),
        bins: vec![0, 9],
        width: 1,
        groups: None,
    });
    fitting::bootstrap(state, model.store(KIND).unwrap(), &config).unwrap();
    let candidates = state.trees[0].candidates[0].clone();
    // Three sampled thresholds, both directions, one missing-only boundary, and incumbent if absent.
    assert!((7..=8).contains(&candidates.len()));
    assert!(candidates.contains(&Candidate {
        feature: 0,
        bin: 4,
        missing_left: true
    }));
    config.split_rank_tolerance = 1.0;
    fitting::bootstrap(state, model.store(KIND).unwrap(), &config).unwrap();
    assert_eq!(state.trees[0].candidates[0], candidates);
    config.split_sampling_rate = 1.0;
    fitting::bootstrap(state, model.store(KIND).unwrap(), &config).unwrap();
    assert_eq!(state.trees[0].candidates[0].len(), 19);
    state.thresholds[0][8] = f64::MAX;
    state.anchors = Arc::new(Anchors {
        ids: state.anchors.ids.clone(),
        y: state.anchors.y.clone(),
        x: state.anchors.x.clone(),
        bins: vec![0, 8],
        width: 1,
        groups: None,
    });
    fitting::bootstrap(state, model.store(KIND).unwrap(), &config).unwrap();
    // A real MAX bin already represents the missing-only split. Avoid a
    // second candidate that would serialize to the same numeric threshold.
    assert_eq!(state.trees[0].candidates[0].len(), 18);
}

#[test]
fn missing_only_seed_works_with_an_empty_numeric_bin_mapper() {
    for mode in [LearnedUpdateMode::LeafOnly, LearnedUpdateMode::Structural] {
        let mut config = controls(mode, LearnedDerivativeRefresh::Eager);
        config.split_sampling_rate = 0.01;
        let mut tree = split(0, false);
        tree.threshold[0] = f64::MAX;
        tree.value = vec![0.0, -2.0, 2.0];
        let (mut learner, mut model) = fixture(
            config,
            &[&[0.0], &[f64::NAN]],
            &[1.0, 9.0],
            5.0,
            vec![tree],
            settings(0.5, 0.0),
        );
        let state = learner.stores.get_mut(&KIND).unwrap();
        state.thresholds = vec![vec![]];
        state.offsets = vec![0, 2];
        state.anchors = Arc::new(Anchors {
            ids: vec![0, 1],
            x: vec![0.0, f64::NAN],
            y: vec![1.0, 9.0],
            bins: vec![0, 1],
            width: 1,
            groups: None,
        });
        fitting::bootstrap(state, model.store(KIND).unwrap(), &learner.config).unwrap();
        checkpoint::validate_seed_fit(state, model.store(KIND).unwrap()).unwrap();
        if mode == LearnedUpdateMode::Structural {
            assert_eq!(
                state.trees[0].candidates[0],
                vec![Candidate {
                    feature: 0,
                    bin: 0,
                    missing_left: false
                }]
            );
        }
        learner
            .observe(KIND, vec![f64::MAX], 1.0, &mut model)
            .unwrap();
        let tree = &model.store(KIND).unwrap().trees[0];
        assert_eq!(tree.threshold[0], f64::MAX);
        assert_eq!(leaf_index(tree, &[f64::MAX]), 1);
        assert_eq!(leaf_index(tree, &[f64::NAN]), 2);
        close(tree.value[1], -2.0);
        close(tree.value[2], 2.0);
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("missing-only");
        learner.save(&path, &model, "{}").unwrap();
        let mut restored_model = checkpoint_model(&path).unwrap();
        let restored =
            OnlineLearner::load(&path, &mut restored_model, &learner.config, &canonical()).unwrap();
        assert_eq!(restored_model.artifact(), model.artifact());
        assert_eq!(
            state_bytes(&restored.stores[&KIND]),
            state_bytes(&learner.stores[&KIND])
        );
    }
}

#[test]
fn structural_rebuild_can_choose_a_missing_only_boundary() {
    let (mut learner, mut model) = fixture(
        controls(
            LearnedUpdateMode::Structural,
            LearnedDerivativeRefresh::Lazy,
        ),
        &[&[0.0], &[1.0], &[f64::NAN]],
        &[1.0, 1.0, 9.0],
        0.0,
        vec![split(0, true)],
        settings(1.0, 0.0),
    );
    learner
        .observe(KIND, vec![f64::MAX], 1.0, &mut model)
        .unwrap();
    let tree = &model.store(KIND).unwrap().trees[0];
    assert_eq!(tree.threshold[0], f64::MAX);
    assert!(!tree.missing_left[0]);
    close(tree.value[tree.left[0] as usize], 1.0);
    close(tree.value[tree.right[0] as usize], 9.0);
    assert_eq!(learner.stores[&KIND].counters.lazy_fallbacks, 1);
    assert!(learner.stores[&KIND].counters.rebuilt_nodes > 0);
}

#[test]
fn compact_candidates_preserve_indices_and_reject_truncation() {
    assert_eq!(std::mem::size_of::<Candidate>(), 6);
    let candidate = Candidate::new(usize::from(u16::MAX), 255, true).unwrap();
    assert_eq!(usize::from(candidate.feature), usize::from(u16::MAX));
    assert_eq!(usize::from(candidate.bin), 255);
    assert!(candidate.missing_left);
    assert!(Candidate::new(usize::from(u16::MAX) + 1, 0, false).is_err());
    assert!(Candidate::new(0, usize::from(u16::MAX) + 1, false).is_err());
}

#[test]
fn weighted_group_mass_matches_hand_computed_leaf_with_regularization() {
    // H={2,4}, R={8}, beta=.25, total mass=3: row weights .375,.375,2.25.
    // G=-20.25; Hessian=3. With lambda=2 and learning rate=.5, leaf=2.025.
    for refresh in [
        LearnedDerivativeRefresh::Eager,
        LearnedDerivativeRefresh::Lazy,
    ] {
        let mut config = controls(LearnedUpdateMode::LeafOnly, refresh);
        config.history_weight = Some(0.25);
        let (mut learner, mut model) = fixture(
            config,
            &[&[0.0], &[1.0]],
            &[2.0, 4.0],
            0.0,
            vec![leaf(0.0)],
            settings(0.5, 2.0),
        );
        learner.observe(KIND, vec![0.0], 8.0, &mut model).unwrap();
        close(model.store(KIND).unwrap().trees[0].value[0], 2.025);
        let stats = learner.stores[&KIND].trees[0].stats[0].total;
        assert_eq!(stats.count, 3);
        assert_eq!(stats.recent_count, 1);
        close(stats.g, -14.0);
        close(stats.recent_g, -8.0);
        checkpoint::validate_state(&learner, &model).unwrap();
    }
}

#[test]
fn weighted_history_duplication_preserves_group_loss_without_regularization() {
    for copies in [1, 7] {
        let mut config = controls(LearnedUpdateMode::LeafOnly, LearnedDerivativeRefresh::Eager);
        config.history_weight = Some(0.25);
        let xs = vec![&[0.0][..]; 2 * copies];
        let ys: Vec<_> = (0..copies).flat_map(|_| [2.0, 4.0]).collect();
        let (mut learner, mut model) =
            fixture(config, &xs, &ys, 0.0, vec![leaf(0.0)], settings(1.0, 0.0));
        learner.observe(KIND, vec![0.0], 8.0, &mut model).unwrap();
        // .25 * mean(H) + .75 * mean(R) = .25*3+.75*8=6.75.
        close(model.store(KIND).unwrap().trees[0].value[0], 6.75);
    }
}

#[test]
fn eager_weighted_downstream_derivatives_use_updated_earlier_trees() {
    let mut config = controls(LearnedUpdateMode::LeafOnly, LearnedDerivativeRefresh::Eager);
    config.history_weight = Some(0.25);
    let (mut learner, mut model) = fixture(
        config,
        &[&[0.0], &[1.0]],
        &[2.0, 4.0],
        0.0,
        vec![leaf(0.0), leaf(0.0)],
        settings(0.5, 0.0),
    );
    learner.observe(KIND, vec![0.0], 8.0, &mut model).unwrap();
    let trees = &model.store(KIND).unwrap().trees;
    // Weighted target mean=6.75. First stage learns half, second half the residual.
    close(trees[0].value[0], 3.375);
    close(trees[1].value[0], 1.6875);
}

#[test]
fn weighted_lazy_evictions_ramp_and_age_match_direct_eager_single_tree() {
    {
        for ramp in [0, 5] {
            let mut config = controls(LearnedUpdateMode::LeafOnly, LearnedDerivativeRefresh::Lazy);
            config.history_weight = Some(0.1);
            config.history_weight_ramp_observations = ramp;
            config.recent_window = 4;
            config.recent_max_age_observations = Some(7);
            config.eager_refresh_interval = 100;
            let (mut lazy, mut lazy_model) = fixture(
                config.clone(),
                &[&[0.0], &[1.0]],
                &[2.0, 4.0],
                0.0,
                vec![split(0, true)],
                settings(0.5, 0.2),
            );
            config.derivative_refresh = LearnedDerivativeRefresh::Eager;
            let (mut eager, mut eager_model) = fixture(
                config,
                &[&[0.0], &[1.0]],
                &[2.0, 4.0],
                0.0,
                vec![split(0, true)],
                settings(0.5, 0.2),
            );
            for i in 0..30 {
                let x = if i % 3 == 0 { 1.0 } else { 0.0 };

                let y = 2.0 + (i % 7) as f64;
                lazy.observe(KIND, vec![x], y, &mut lazy_model).unwrap();
                eager.observe(KIND, vec![x], y, &mut eager_model).unwrap();
                assert_eq!(
                    lazy.stores[&KIND]
                        .recent
                        .iter()
                        .map(|r| r.id)
                        .collect::<Vec<_>>(),
                    eager.stores[&KIND]
                        .recent
                        .iter()
                        .map(|r| r.id)
                        .collect::<Vec<_>>()
                );
                for (a, b) in lazy_model.store(KIND).unwrap().trees[0]
                    .value
                    .iter()
                    .zip(&eager_model.store(KIND).unwrap().trees[0].value)
                {
                    close(*a, *b);
                }
                checkpoint::validate_state(&lazy, &lazy_model).unwrap();
            }
            assert!(lazy.stores[&KIND].counters.lazy_updates > 0);
        }
    }
}

#[test]
fn mutable_tail_and_update_step_preserve_frozen_seed_and_convex_leaf_displacement() {
    let mut config = controls(LearnedUpdateMode::LeafOnly, LearnedDerivativeRefresh::Lazy);
    config.history_weight = Some(0.5);
    config.mutable_tree_tail = Some(1);
    config.update_step = 0.25;
    let (mut learner, mut model) = fixture(
        config,
        &[&[0.0], &[1.0]],
        &[12.0, 12.0],
        10.0,
        vec![leaf(1.0), leaf(2.0), leaf(3.0)],
        settings(1.0, 0.0),
    );
    let original = model.artifact().clone();
    assert!(!learner.flush(KIND, &mut model).unwrap());
    assert_eq!(model.artifact(), original);
    learner.observe(KIND, vec![0.0], 28.0, &mut model).unwrap();
    let trees = &model.store(KIND).unwrap().trees;
    assert_eq!(trees.len(), 3);
    close(trees[0].value[0], 1.0);
    close(trees[1].value[0], 2.0);
    // Mean target=20, frozen prefix=13, fitted final leaf=7. Step:3+.25*(7-3)=4.
    close(trees[2].value[0], 4.0);
    checkpoint::validate_state(&learner, &model).unwrap();
}

fn structural_tail_fixture(
    mut config: FpmLearnedConfig,
    learning_rate: f64,
) -> (OnlineLearner, LearnedForwardPassModel) {
    config.mutable_tree_tail = Some(1);
    let mut first = split(0, true);
    first.value[1..].fill(1.0);
    let mut second = split(1, true);
    second.value[1..].fill(2.0);
    fixture(
        config,
        &[&[0.0, 0.0], &[0.0, 1.0], &[1.0, 0.0], &[1.0, 1.0]],
        &[14.0, 22.0, 14.0, 22.0],
        10.0,
        vec![first, second, split(0, true)],
        settings(learning_rate, 0.0),
    )
}

fn structural_tree_allocations(tree: &TreeState) -> Vec<usize> {
    // Empty Vec pointers cannot establish allocation identity. This fixture
    // must exercise the structural buffers that freezing a prefix should retain.
    assert!(!tree.stats.is_empty());
    assert!(tree.stats.iter().any(|node| !node.histogram.is_empty()));
    assert!(
        tree.candidates
            .iter()
            .any(|candidates| !candidates.is_empty())
    );
    assert!(tree.regions.iter().all(|region| !region.is_empty()));
    let mut pointers = vec![
        tree.stats.as_ptr() as usize,
        tree.candidates.as_ptr() as usize,
        tree.regions.as_ptr() as usize,
    ];
    pointers.extend(
        tree.stats
            .iter()
            .filter(|node| !node.histogram.is_empty())
            .map(|node| node.histogram.as_ptr() as usize),
    );
    pointers.extend(
        tree.candidates
            .iter()
            .filter(|candidates| !candidates.is_empty())
            .map(|candidates| candidates.as_ptr() as usize),
    );
    pointers.extend(tree.regions.iter().map(|region| region.as_ptr() as usize));
    pointers
}

fn fixture_prediction(model: &LearnedForwardPassModel, x: &[f64]) -> f64 {
    use super::super::learned::{IterationFeatureVector, feature_names};
    let names = feature_names();
    let mut features = IterationFeatureVector {
        values: vec![f64::NAN; names.len()],
        request_lists: false,
    };
    for (name, value) in model.feature_names().iter().zip(x) {
        features.values[names.iter().position(|entry| entry == name).unwrap()] = *value;
    }
    model.predict_ms(KIND, &features).unwrap().unwrap()
}

#[test]
fn mutable_tail_eager_publication_preserves_frozen_structural_allocations() {
    for (refresh, interval, expected_eager, expected_lazy, expected_fallbacks) in [
        (LearnedDerivativeRefresh::Eager, 32, 3, 0, 0),
        (LearnedDerivativeRefresh::Lazy, 1, 3, 0, 0),
        (LearnedDerivativeRefresh::Lazy, 2, 2, 1, 1),
    ] {
        let mut config = controls(LearnedUpdateMode::Structural, refresh);
        config.eager_refresh_interval = interval;
        let (mut learner, mut model) = structural_tail_fixture(config, 1.0);
        let frozen_bytes = bincode::serialize(&learner.stores[&KIND].trees[..2]).unwrap();
        let frozen_allocations: Vec<_> = learner.stores[&KIND].trees[..2]
            .iter()
            .map(structural_tree_allocations)
            .collect();
        let frozen_predictors = model.store(KIND).unwrap().trees[..2].to_vec();
        let mut previous_tail = model.store(KIND).unwrap().trees[2].clone();
        // Frozen prefix = 10 + 1 + 2 = 13. The optimal tail splits on feature 1.
        // Left residuals are [1, 1, 1], then [1, 1, 1, 5], then [1, 1, 1, 5, 9],
        // giving means 1, 2, 3.4; its two right residuals remain 9 throughout.
        for (label, expected_left) in [(14.0, 14.0), (18.0, 15.0), (22.0, 16.4)] {
            learner
                .observe(KIND, vec![0.0, 0.0], label, &mut model)
                .unwrap();
            let state = &learner.stores[&KIND];
            assert_eq!(bincode::serialize(&state.trees[..2]).unwrap(), frozen_bytes);
            assert_eq!(
                state.trees[..2]
                    .iter()
                    .map(structural_tree_allocations)
                    .collect::<Vec<_>>(),
                frozen_allocations,
            );
            let trees = &model.store(KIND).unwrap().trees;
            assert_eq!(&trees[..2], frozen_predictors.as_slice());
            assert_ne!(trees[2], previous_tail);
            assert_eq!(trees[2].feature[0], 1);
            close(fixture_prediction(&model, &[1.0, 0.0]), expected_left);
            close(fixture_prediction(&model, &[1.0, 1.0]), 22.0);
            previous_tail = trees[2].clone();
            checkpoint::validate_state(&learner, &model).unwrap();
        }
        let counters = &learner.stores[&KIND].counters;
        assert_eq!(counters.eager_updates, expected_eager);
        assert_eq!(counters.lazy_updates, expected_lazy);
        assert_eq!(counters.lazy_fallbacks, expected_fallbacks);
        assert!(counters.rebuilt_nodes > 0);
    }
}

#[test]
fn mutable_tail_prediction_range_failure_preserves_allocations_and_retries() {
    for (refresh, interval) in [
        (LearnedDerivativeRefresh::Eager, 32),
        (LearnedDerivativeRefresh::Lazy, 1),
        (LearnedDerivativeRefresh::Lazy, 32),
    ] {
        let mut config = controls(LearnedUpdateMode::Structural, refresh);
        config.update_every = 2;
        config.eager_refresh_interval = interval;
        let (mut learner, model) = structural_tail_fixture(config, 100.0);
        let mut artifact = model.artifact();
        artifact.target = LearnedTarget::LogMs;
        let mut model = LearnedForwardPassModel::from_artifact(artifact).unwrap();
        learner
            .observe(KIND, vec![0.0, 0.0], 14.0_f64.exp(), &mut model)
            .unwrap();
        let before_state = state_bytes(&learner.stores[&KIND]);
        let before_model = model.artifact();
        let allocations: Vec<_> = learner.stores[&KIND]
            .trees
            .iter()
            .map(structural_tree_allocations)
            .collect();
        // All fitting arithmetic is finite: right residual 9 produces leaf 900
        // at eta 100. Final link-range validation rejects raw upper 13 + 900 = 913.
        let error = learner.flush(KIND, &mut model).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("updated prediction range can overflow")
        );
        let state = &learner.stores[&KIND];
        assert_eq!(state_bytes(state), before_state);
        assert_eq!(model.artifact(), before_model);
        assert_eq!(
            state
                .trees
                .iter()
                .map(structural_tree_allocations)
                .collect::<Vec<_>>(),
            allocations,
        );
        assert_eq!(learner.version, 0);
        assert_eq!(state.pending.len(), 1);
        assert!(state.recent.is_empty());
        assert_eq!(state.counters.accepted, 1);
        assert_eq!(state.counters.failures, 1);
        assert_eq!(state.counters.publications, 0);

        learner
            .stores
            .get_mut(&KIND)
            .unwrap()
            .settings
            .learning_rate = 1.0;
        assert!(learner.flush(KIND, &mut model).unwrap());
        assert_eq!(learner.version, 1);
        assert!(learner.stores[&KIND].pending.is_empty());
        assert_eq!(learner.stores[&KIND].recent.len(), 1);
        assert_eq!(
            learner.stores[&KIND].trees[..2]
                .iter()
                .map(structural_tree_allocations)
                .collect::<Vec<_>>(),
            allocations[..2],
        );
        close(fixture_prediction(&model, &[1.0, 0.0]).ln(), 14.0);
        close(fixture_prediction(&model, &[1.0, 1.0]).ln(), 22.0);
        checkpoint::validate_state(&learner, &model).unwrap();
    }
}

#[test]
fn weighted_configuration_round_trip_and_invalid_values() {
    let mut config = controls(LearnedUpdateMode::LeafOnly, LearnedDerivativeRefresh::Lazy);
    config.history_weight = Some(0.1);
    config.history_weight_ramp_observations = 2048;
    config.mutable_tree_tail = Some(5);
    config.update_step = 0.5;
    config.recent_max_age_observations = Some(4096);
    config.validate().unwrap();
    let json = serde_json::to_string(&config).unwrap();
    assert_eq!(
        serde_json::from_str::<FpmLearnedConfig>(&json).unwrap(),
        config
    );
    for mutation in [0, 1, 2, 3, 4, 5, 6] {
        let mut bad = config.clone();
        match mutation {
            0 => bad.history_weight = Some(-0.1),
            1 => bad.history_weight = Some(f64::NAN),
            2 => bad.history_weight = None,
            3 => bad.mutable_tree_tail = Some(0),
            4 => bad.update_step = 0.0,
            5 => bad.recent_max_age_observations = Some(0),
            _ => bad.update_mode = LearnedUpdateMode::Structural,
        }
        assert!(bad.validate().is_err());
    }
}

#[test]
fn explicit_unit_group_masses_match_existing_unweighted_oracle() {
    for refresh in [
        LearnedDerivativeRefresh::Eager,
        LearnedDerivativeRefresh::Lazy,
    ] {
        let config = controls(LearnedUpdateMode::LeafOnly, refresh);
        let (mut unit, mut unit_model) = fixture(
            config.clone(),
            &[&[0.0], &[1.0]],
            &[2.0, 4.0],
            1.0,
            vec![leaf(0.0)],
            settings(0.2, 3.0),
        );
        let mut weighted_config = config;
        weighted_config.history_weight = Some(2.0 / 3.0);
        let (mut weighted, mut weighted_model) = fixture(
            weighted_config,
            &[&[0.0], &[1.0]],
            &[2.0, 4.0],
            1.0,
            vec![leaf(0.0)],
            settings(0.2, 3.0),
        );
        unit.observe(KIND, vec![0.0], 8.0, &mut unit_model).unwrap();
        weighted
            .observe(KIND, vec![0.0], 8.0, &mut weighted_model)
            .unwrap();
        close(
            unit_model.store(KIND).unwrap().trees[0].value[0],
            0.2 * 11.0 / 6.0,
        );
        close(
            unit_model.store(KIND).unwrap().trees[0].value[0],
            weighted_model.store(KIND).unwrap().trees[0].value[0],
        );
    }
}

#[test]
fn weighted_structural_stump_matches_raw_regularized_loss_oracle() {
    // Independent oracle enumerates raw partitions, minimizing the weighted
    // squared loss plus L2 directly. It never uses production histograms/gains.
    let xs = [
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![f64::NAN, 0.0],
        vec![f64::NAN, 1.0],
        vec![0.0, 0.0],
        vec![1.0, 1.0],
    ];
    let ys = [1.0, 2.0, 9.0, 8.0, 1.5, 9.5, 3.0, 14.0];
    for beta in [0.1, 0.5, 0.9] {
        for ridge in [0.0, 0.5, 2.0] {
            for refresh in [
                LearnedDerivativeRefresh::Eager,
                LearnedDerivativeRefresh::Lazy,
            ] {
                let mut config = controls(LearnedUpdateMode::Structural, refresh);
                config.history_weight = Some(beta);
                config.update_every = 2;
                let mut params = settings(0.4, ridge);
                params.min_samples_leaf = 2;
                let (mut learner, mut model) = fixture(
                    config,
                    &xs[..6].iter().map(Vec::as_slice).collect::<Vec<_>>(),
                    &ys[..6],
                    5.0,
                    vec![split(0, true)],
                    params,
                );
                for i in 6..8 {
                    learner
                        .observe(KIND, xs[i].clone(), ys[i], &mut model)
                        .unwrap();
                }
                let weights: Vec<_> = (0..8)
                    .map(|i| {
                        if i < 6 {
                            8.0 * beta / 6.0
                        } else {
                            8.0 * (1.0 - beta) / 2.0
                        }
                    })
                    .collect();
                let objective = |members: &[usize]| {
                    let h = members.iter().map(|&i| weights[i]).sum::<f64>();
                    let a = members
                        .iter()
                        .map(|&i| weights[i] * (ys[i] - 5.0))
                        .sum::<f64>()
                        / (h + ridge);
                    let loss = members
                        .iter()
                        .map(|&i| 0.5 * weights[i] * (5.0 + a - ys[i]).powi(2))
                        .sum::<f64>()
                        + 0.5 * ridge * a * a;
                    (loss, a)
                };
                let mut optimum = f64::INFINITY;
                for feature in 0..2 {
                    for (threshold, missing) in [(0.5, false), (0.5, true), (f64::MAX, false)] {
                        let (left, right): (Vec<_>, Vec<_>) = (0..8).partition(|&i| {
                            if xs[i][feature].is_nan() {
                                missing
                            } else {
                                xs[i][feature] <= threshold
                            }
                        });
                        if left.len() < 2 || right.len() < 2 {
                            continue;
                        }
                        optimum = optimum.min(objective(&left).0 + objective(&right).0);
                    }
                }
                let tree = &model.store(KIND).unwrap().trees[0];
                assert!(tree.left[0] >= 0);
                let (left, right): (Vec<_>, Vec<_>) =
                    (0..8).partition(|&i| leaf_index(tree, &xs[i]) == tree.left[0] as usize);
                assert!(left.len() >= 2 && right.len() >= 2);
                let (ll, la) = objective(&left);
                let (rl, ra) = objective(&right);
                close(ll + rl, optimum);
                close(tree.value[tree.left[0] as usize], 0.4 * la);
                close(tree.value[tree.right[0] as usize], 0.4 * ra);
                checkpoint::validate_state(&learner, &model).unwrap();
            }
        }
    }
}

#[test]
fn weighted_tail_runtime_checkpoint_continues_groups_ramp_and_pending_exactly() {
    let mut config = controls(LearnedUpdateMode::LeafOnly, LearnedDerivativeRefresh::Lazy);
    config.update_every = 2;
    config.history_weight = Some(0.1);
    config.history_weight_ramp_observations = 8;
    config.mutable_tree_tail = Some(1);
    config.update_step = 0.5;
    config.recent_max_age_observations = Some(5);
    let (mut learner, mut model) = fixture(
        config,
        &[&[0.0], &[1.0]],
        &[1.0, 3.0],
        1.0,
        vec![leaf(1.0), leaf(0.0)],
        settings(0.4, 0.2),
    );
    for i in 0..9 {
        learner
            .observe(KIND, vec![(i % 2) as f64], 2.0 + i as f64, &mut model)
            .unwrap();
    }
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("weighted");
    learner.save(&path, &model, "{}").unwrap();
    let mut resumed_model = checkpoint_model(&path).unwrap();
    let mut resumed =
        OnlineLearner::load(&path, &mut resumed_model, &learner.config, &canonical()).unwrap();
    assert_eq!(
        state_bytes(&learner.stores[&KIND]),
        state_bytes(&resumed.stores[&KIND])
    );
    for i in 9..19 {
        learner
            .observe(KIND, vec![(i % 2) as f64], 2.0 + i as f64, &mut model)
            .unwrap();
        resumed
            .observe(
                KIND,
                vec![(i % 2) as f64],
                2.0 + i as f64,
                &mut resumed_model,
            )
            .unwrap();
    }
    assert_eq!(model.artifact(), resumed_model.artifact());
    assert_eq!(
        state_bytes(&learner.stores[&KIND]),
        state_bytes(&resumed.stores[&KIND])
    );
    let manifest_path = path.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    assert_eq!(manifest["schema_version"], 5);
    manifest["schema_version"] = serde_json::json!(2);
    std::fs::write(manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(
        checkpoint_model(&path)
            .unwrap_err()
            .to_string()
            .contains("version 2 predates weighted group statistics")
    );
}

#[test]
fn runtime_source_paths_and_manifest_hash_can_change_without_changing_fitting_controls() {
    let config = controls(LearnedUpdateMode::LeafOnly, LearnedDerivativeRefresh::Eager);
    let (mut learner, mut model) = fixture(
        config,
        &[&[0.0], &[1.0]],
        &[1.0, 3.0],
        2.0,
        vec![leaf(0.0)],
        settings(0.5, 0.0),
    );
    learner.config.source_sha256 = Some("a".repeat(64));
    learner.observe(KIND, vec![1.0], 7.0, &mut model).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let first = directory.path().join("first");
    learner.save(&first, &model, "{}").unwrap();
    let moved = directory.path().join("moved");
    std::fs::rename(&first, &moved).unwrap();
    let mut relocated = learner.config.clone();
    relocated.checkpoint_path = Some(moved.clone());
    {
        use sha2::{Digest, Sha256};
        relocated.source_sha256 = Some(format!(
            "{:x}",
            Sha256::digest(std::fs::read(moved.join("manifest.json")).unwrap())
        ));
    }
    let mut restored_model = checkpoint_model(&moved).unwrap();
    let restored =
        OnlineLearner::load(&moved, &mut restored_model, &relocated, &canonical()).unwrap();
    assert_eq!(
        state_bytes(&learner.stores[&KIND]),
        state_bytes(&restored.stores[&KIND])
    );
    assert_eq!(model.artifact(), restored_model.artifact());
    relocated.update_every += 1;
    assert!(
        OnlineLearner::load(&moved, &mut restored_model, &relocated, &canonical())
            .unwrap_err()
            .to_string()
            .contains("online controls differ")
    );
}

#[test]
fn research_v4_runtime_layout_is_rejected_before_reading_binary_state() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("manifest.json"),
        serde_json::to_vec(&serde_json::json!({
            "schema":"aisimulate_fpm_online_checkpoint", "schema_version":4,
            "artifact_file":"model.json", "artifact_sha256":"unused", "identity":{}
        }))
        .unwrap(),
    )
    .unwrap();
    assert!(
        checkpoint_model(dir.path())
            .unwrap_err()
            .to_string()
            .contains("version 4 uses the research replay/configuration layout")
    );
}
