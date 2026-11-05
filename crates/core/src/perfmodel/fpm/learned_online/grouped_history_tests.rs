// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn compact(learner: &mut OnlineLearner, model: &LearnedForwardPassModel) {
    learner.config.history_bin_grouping = true;
    for (kind, state) in &mut learner.stores {
        state.anchors = Arc::new(state.anchors.group_by_bins().unwrap());
        fitting::cache_frozen_anchor_prefix(state, model.store(*kind).unwrap(), &learner.config);
    }
}

fn near(a: f64, b: f64) {
    assert!(
        (a - b).abs() <= 2e-11 * a.abs().max(b.abs()).max(1.0),
        "{a} != {b}"
    );
}

fn same_predictor(a: &LearnedForwardPassModel, b: &LearnedForwardPassModel) {
    let a = a.store(KIND).unwrap();
    let b = b.store(KIND).unwrap();
    assert_eq!(a.baseline, b.baseline);
    assert_eq!(a.trees.len(), b.trees.len());
    for (a, b) in a.trees.iter().zip(&b.trees) {
        assert_eq!(a.left, b.left);
        assert_eq!(a.right, b.right);
        assert_eq!(a.feature, b.feature);
        assert_eq!(a.threshold, b.threshold);
        assert_eq!(a.missing_left, b.missing_left);
        for (&a, &b) in a.value.iter().zip(&b.value) {
            near(a, b);
        }
    }
}

#[test]
fn complete_bins_combine_different_targets_and_keep_missing_values_separate() {
    // Three observations in bin [0,0] have latencies 10,20,40 ms.
    // Their hand-derived target sum is ln(8000), and their gradient at
    // ln(30) is ln(30/10)+ln(30/20)+ln(30/40) = ln(27/8).
    let thresholds = vec![vec![8.0], vec![1000.0]];
    let xs = [
        [2.0, 200.0],
        [4.0, 400.0],
        [7.0, 800.0],
        [4.0, 1400.0],
        [f64::NAN, 200.0],
        [f64::NAN, 400.0],
    ];
    let anchors = Anchors {
        ids: vec![4, 8, 9, 11, 13, 18],
        x: xs.iter().flatten().copied().collect(),
        y: [10.0_f64, 20.0, 40.0, 30.0, 5.0, 10.0]
            .iter()
            .map(|x| x.ln())
            .collect(),
        bins: xs.iter().flat_map(|x| bin_row(x, &thresholds)).collect(),
        width: 2,
        groups: None,
    };
    let grouped = anchors.group_by_bins().unwrap();
    assert_eq!(grouped.ids, vec![4, 11, 13]);
    assert_eq!(grouped.groups.as_ref().unwrap().counts, vec![3, 1, 2]);
    assert_eq!(grouped.original_rows(), 6);
    assert_eq!(grouped.groups.as_ref().unwrap().original_max_id, 18);
    assert_eq!(grouped.bins, vec![0, 0, 0, 1, 2, 0]);
    near(grouped.y[0], 8000.0_f64.ln());
    let rows = Rows {
        anchors: &grouped,
        recent: vec![],
    };
    near(rows.gradient(0, 30.0_f64.ln()), (27.0_f64 / 8.0).ln());
    assert_eq!(rows.count(0), 3);
    assert!(grouped.group_by_bins().is_err());
    assert_eq!(
        bincode::serialize(&grouped).unwrap(),
        bincode::serialize(&anchors.group_by_bins().unwrap()).unwrap()
    );
}

#[test]
fn grouped_eager_and_lazy_updates_preserve_structures_predictions_and_logical_counts() {
    for mode in [LearnedUpdateMode::LeafOnly, LearnedUpdateMode::Structural] {
        for refresh in [
            LearnedDerivativeRefresh::Eager,
            LearnedDerivativeRefresh::Lazy,
        ] {
            for beta in [None, Some(0.1)] {
                let xs: Vec<Vec<f64>> = (0..100)
                    .map(|i| match i % 5 {
                        0 => vec![0.1, 0.1],
                        1 => vec![0.3, 0.8],
                        2 => vec![0.8, 0.2],
                        3 => vec![0.9, 0.9],
                        _ => vec![f64::NAN, 0.7],
                    })
                    .collect();
                let refs: Vec<_> = xs.iter().map(Vec::as_slice).collect();
                let ys: Vec<_> = (0..100).map(|i| (3.0 + (i % 7) as f64).ln()).collect();
                let mut config = controls(mode, refresh);
                config.history_weight = beta;
                config.history_weight_ramp_observations = if beta.is_some() { 4 } else { 0 };
                config.eager_refresh_interval = 4;
                config.split_rank_tolerance = 0.01;
                let mut fitting_settings = settings(0.2, 0.3);
                fitting_settings.min_samples_leaf = 5;
                let (mut baseline, model) = fixture(
                    config,
                    &refs,
                    &ys,
                    1.5,
                    vec![split(0, true), split(1, false), split(0, false)],
                    fitting_settings,
                );
                let mut artifact = model.artifact();
                artifact.target = LearnedTarget::LogMs;
                let mut model = LearnedForwardPassModel::from_artifact(artifact).unwrap();
                let mut grouped = baseline.clone();
                compact(&mut grouped, &model);
                let mut grouped_model = model.clone();
                assert_eq!(model.artifact(), grouped_model.artifact());
                assert_eq!(grouped.stores[&KIND].anchors.y.len(), 5);
                for i in 0..30 {
                    let x = xs[(i * 3) % xs.len()].clone();
                    let wall = 2.0 + (i * 7 % 19) as f64;
                    baseline.observe(KIND, x.clone(), wall, &mut model).unwrap();
                    grouped.observe(KIND, x, wall, &mut grouped_model).unwrap();
                    same_predictor(&model, &grouped_model);
                    checkpoint::validate_state(&grouped, &grouped_model).unwrap();
                    for (a, b) in baseline.stores[&KIND]
                        .trees
                        .iter()
                        .zip(&grouped.stores[&KIND].trees)
                    {
                        assert_eq!(a.generation, b.generation);
                        for (a, b) in a.stats.iter().zip(&b.stats) {
                            assert_eq!(a.total.count, b.total.count);
                            assert_eq!(a.total.recent_count, b.total.recent_count);
                            near(a.total.g, b.total.g);
                        }
                    }
                }
                let diagnostics = grouped.diagnostics();
                let store = &diagnostics["stores"]["pure_decode"];
                assert_eq!(store["anchor_rows"], 100);
                assert_eq!(store["historical_storage_rows"], 5);
                assert_eq!(store["history_compression_ratio"], 20.0);
            }
        }
    }
}

#[test]
fn grouped_runtime_resume_preserves_pending_rows_ramp_tail_and_next_id() {
    let mut config = controls(LearnedUpdateMode::LeafOnly, LearnedDerivativeRefresh::Lazy);
    config.history_weight = Some(0.1);
    config.history_weight_ramp_observations = 8;
    config.update_every = 2;
    config.mutable_tree_tail = Some(1);
    config.eager_refresh_interval = 3;
    let (mut learner, mut model) = fixture(
        config,
        &[&[0.1], &[0.2], &[0.9], &[0.8]],
        &[2.0, 4.0, 5.0, 9.0],
        1.0,
        vec![leaf(0.5), split(0, true)],
        settings(0.2, 0.3),
    );
    compact(&mut learner, &model);
    for i in 0..9 {
        learner
            .observe(KIND, vec![(i % 2) as f64], 2.0 + i as f64, &mut model)
            .unwrap();
    }
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("grouped");
    learner.save(&path, &model, "{}").unwrap();
    let mut restored_model = checkpoint_model(&path).unwrap();
    let mut restored =
        OnlineLearner::load(&path, &mut restored_model, &learner.config, &canonical()).unwrap();
    assert_eq!(learner.next_row_id, restored.next_row_id);
    assert_eq!(
        state_bytes(&learner.stores[&KIND]),
        state_bytes(&restored.stores[&KIND])
    );
    for i in 9..25 {
        learner
            .observe(KIND, vec![(i % 2) as f64], 2.0 + i as f64, &mut model)
            .unwrap();
        restored
            .observe(
                KIND,
                vec![(i % 2) as f64],
                2.0 + i as f64,
                &mut restored_model,
            )
            .unwrap();
        assert_eq!(model.artifact(), restored_model.artifact());
        checkpoint::validate_state(&restored, &restored_model).unwrap();
    }
    assert_eq!(
        state_bytes(&learner.stores[&KIND]),
        state_bytes(&restored.stores[&KIND])
    );
    let mut config = learner.config.clone();
    config.history_bin_grouping = false;
    let mut checkpoint_predictor = checkpoint_model(&path).unwrap();
    assert!(
        OnlineLearner::load(&path, &mut checkpoint_predictor, &config, &canonical())
            .unwrap_err()
            .to_string()
            .contains("controls differ")
    );
}

#[test]
fn grouped_runtime_validation_rejects_corrupt_multiplicity_and_duplicate_bins() {
    let (mut learner, model) = fixture(
        controls(LearnedUpdateMode::LeafOnly, LearnedDerivativeRefresh::Eager),
        &[&[0.0], &[0.1], &[1.0]],
        &[1.0, 2.0, 3.0],
        0.0,
        vec![leaf(0.0)],
        settings(0.2, 0.0),
    );
    compact(&mut learner, &model);
    checkpoint::validate_state(&learner, &model).unwrap();
    // Arc serialization gives independent editable copies without adding Clone
    // to production anchor storage solely for test corruption.
    for corruption in 0..4 {
        let mut broken = learner.clone();
        let anchors = &mut broken.stores.get_mut(&KIND).unwrap().anchors;
        let mut data: Anchors =
            bincode::deserialize(&bincode::serialize(anchors).unwrap()).unwrap();
        match corruption {
            0 => data.groups.as_mut().unwrap().counts[0] = 0,
            1 => data.groups.as_mut().unwrap().original_rows += 1,
            2 => {
                data.bins[1] = data.bins[0];
                data.x[1] = data.x[0];
            }
            _ => data.groups.as_mut().unwrap().original_max_id = broken.next_row_id,
        }
        *anchors = Arc::new(data);
        assert!(checkpoint::validate_state(&broken, &model).is_err());
    }
}

#[test]
fn grouping_control_roundtrips_and_legacy_runtime_schema_is_explicitly_rejected() {
    let default = FpmLearnedConfig::default();
    assert!(default.history_bin_grouping);
    let config: FpmLearnedConfig =
        serde_json::from_value(serde_json::json!({"history_bin_grouping":false})).unwrap();
    assert!(!config.history_bin_grouping);
    assert_eq!(
        serde_json::from_str::<FpmLearnedConfig>(&serde_json::to_string(&config).unwrap()).unwrap(),
        config
    );
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("manifest.json"),
        serde_json::to_vec(&serde_json::json!({
            "schema":"aisimulate_fpm_online_checkpoint", "schema_version":3,
            "artifact_file":"model.json", "artifact_sha256":"unused", "identity":{}
        }))
        .unwrap(),
    )
    .unwrap();
    assert!(
        checkpoint_model(dir.path())
            .unwrap_err()
            .to_string()
            .contains("version 3 predates historical bin grouping")
    );
}
