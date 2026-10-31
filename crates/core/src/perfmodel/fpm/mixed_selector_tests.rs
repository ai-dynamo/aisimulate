// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::perfmodel::fpm::model::combined::tests::checkpoint_with_stores;

// Manifest validation is covered above and in the public SDK tests. These
// dispatch tests attach the independent hand-oracle direct fixture to a real
// two-pure-store training checkpoint, then append a real one-store C checkpoint.
fn selector() -> (
    Fixture,
    tempfile::TempDir,
    tempfile::TempDir,
    ForwardPassPerfModel,
) {
    let mut fixture = Fixture::new(0, 0, true);
    fixture
        .config
        .estimator_config
        .temporary_mixed_composition
        .as_mut()
        .unwrap()
        .calibration
        .factor_bounds
        .max = Some(2.0);
    let (seed, config) = checkpoint_with_stores(
        ForwardPassWorkerType::Aggregated,
        &[Kind::PureDecode, Kind::PurePrefill],
        2,
        10.0,
    );
    let table = tempfile::tempdir().unwrap();
    let mut model = ForwardPassPerfModel::best_available(config).unwrap();
    model.mixed_composition = Some(fixture.runtime(table.path()));
    (fixture, seed, table, model)
}

fn seed(kind: Kind, count: usize) -> tempfile::TempDir {
    checkpoint_with_stores(ForwardPassWorkerType::Aggregated, &[kind], count, 40.0).0
}

fn gate(model: &ForwardPassPerfModel, kind: Kind) -> serde_json::Value {
    model.learned_selector_diagnostics().unwrap()["states"]
        .as_array()
        .unwrap()
        .iter()
        .find(|state| {
            state["store"] == serde_json::to_value(kind).unwrap() && state["query_type"] == "mixed"
        })
        .unwrap()
        .clone()
}

#[test]
fn mixed_selector_rejects_incomplete_portable_snapshot_before_writing() {
    let (_fixture, _a, _table, model) = selector();
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("snapshot");
    let before = model.mixed_composition_state().unwrap();
    let error = model
        .save_state_snapshot(&destination, &Default::default())
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("not supported by portable state snapshots")
    );
    assert!(!destination.exists());
    assert_eq!(model.mixed_composition_state().unwrap(), before);
}

#[test]
fn mixed_selector_bootstrap_preserves_prior_and_starts_comparisons_after_seed() {
    let (_fixture, _a, _table, mut model) = selector();
    let c = seed(Kind::ContainsLocallyMixed, 20);
    let mut rows = group(Kind::ContainsLocallyMixed);
    rows[0].wall_time = 0.040;
    let stable_before = model.learned_expert_artifacts().unwrap()["stable"].clone();
    // Hand oracle: raw direct composition is 14 ms; factor max=2 caps the
    // calibrated stable expert at 28 ms, despite all bootstrap labels being40.
    for index in 0..20 {
        let prediction = model.estimate_forward_pass_with_route(&rows).unwrap();
        let route = &prediction["temporary_mixed_composition"];
        assert_eq!(route["online_ready"], false);
        assert_eq!(route["selected_expert"], "stable");
        assert_eq!(route["paired_comparisons"], 0);
        assert_eq!(
            prediction["latency_ms"],
            if index < 5 { 14.0 } else { 28.0 }
        );
        model.tune_with_fpms(&[rows.clone()]).unwrap();
    }
    let mixed_before = model.mixed_composition_state().unwrap();
    let gate_before = gate(&model, Kind::ContainsLocallyMixed);
    assert_eq!(model.append_mixed_learned_checkpoint(c.path()).unwrap(), 1);
    assert_eq!(model.mixed_composition_state().unwrap(), mixed_before);
    let after = gate(&model, Kind::ContainsLocallyMixed);
    assert_eq!(after["observations"], gate_before["observations"]);
    assert_eq!(after["stable_source"], "interpolation");
    assert_eq!(after["online_ready"], true);
    let artifacts = model.learned_expert_artifacts().unwrap();
    assert_eq!(artifacts["stable"], stable_before);
    assert!(
        artifacts["stable"]["stores"]
            .get("contains_locally_mixed")
            .is_none()
    );
    assert_eq!(
        artifacts["online"]["stores"]["contains_locally_mixed"]["baseline"],
        40.0
    );
    assert_eq!(
        model.ordinary_accepted_observations(Kind::ContainsLocallyMixed),
        0
    );
    // Existing warmup3/persistence2: C21..C24 query stable, C25 queries online.
    // No bootstrap row contributes a paired selector loss.
    for index in 0..5 {
        let prediction = model.estimate_forward_pass_with_route(&rows).unwrap();
        let route = &prediction["temporary_mixed_composition"];
        assert_eq!(route["paired_comparisons"], index);
        assert_eq!(
            route["selected_expert"],
            if index < 4 { "stable" } else { "online" }
        );
        assert_eq!(
            prediction["latency_ms"],
            if index < 4 { 28.0 } else { 40.0 }
        );
        if index < 4 {
            assert!(route["online_available"].is_null());
        } else {
            assert!(route["stable_available"].is_null());
        }
        let before_gate = model.learned_selector_diagnostics();
        let mut target_changed = rows.clone();
        target_changed[0].wall_time = 999.0;
        assert_eq!(
            model
                .estimate_forward_pass_with_route(&target_changed)
                .unwrap(),
            prediction
        );
        assert_eq!(model.learned_selector_diagnostics(), before_gate);
        model.tune_with_fpms(&[rows.clone()]).unwrap();
    }
    assert_eq!(
        model.ordinary_accepted_observations(Kind::ContainsLocallyMixed),
        5
    );
    assert_eq!(
        model.mixed_composition.as_ref().unwrap().admissions[&Kind::ContainsLocallyMixed],
        25
    );
    assert_eq!(
        model
            .estimate_forward_pass_time_ms(&group(Kind::PureDecode))
            .unwrap(),
        Some(10.0)
    );
    // Once Online is selected, prediction does not invoke prior-only complete
    // rank validation. The chosen forest needs only the scheduled input group.
    assert_eq!(
        model.estimate_forward_pass_time_ms(&rows[..1]).unwrap(),
        Some(40.0)
    );
}

#[test]
fn mixed_selector_route_identifies_critical_rank_query_key() {
    let (_fixture, _a, _table, mut model) = selector();
    let c = seed(Kind::CrossRankAggregated, 20);
    model.append_mixed_learned_checkpoint(c.path()).unwrap();
    // Both groups use the cross-rank mixed forest. The first critical rank is
    // Decode (12 requests + 96 KV > 100 prefill); the second is Prefill (200).
    for (tokens, query_type) in [(100, "decode"), (200, "prefill")] {
        let mut rows = group(Kind::CrossRankAggregated);
        rows[0].scheduled_requests.sum_prefill_tokens = tokens;
        let before = model.estimate_forward_pass_with_route(&rows).unwrap();
        let route = &before["temporary_mixed_composition"];
        assert_eq!(route["workload_kind"], "cross_rank_aggregated");
        assert_eq!(route["query_type"], query_type);
        assert_eq!(route["paired_comparisons"], 0);
        model.tune_with_fpms(&[rows.clone()]).unwrap();
        let after = model.estimate_forward_pass_with_route(&rows).unwrap();
        assert_eq!(
            after["temporary_mixed_composition"]["paired_comparisons"],
            1
        );
        let diagnostics = model.learned_selector_diagnostics().unwrap();
        let state = diagnostics["states"]
            .as_array()
            .unwrap()
            .iter()
            .find(|state| {
                state["store"] == "cross_rank_aggregated" && state["query_type"] == query_type
            })
            .unwrap();
        assert_eq!(
            state["observations"],
            after["temporary_mixed_composition"]["paired_comparisons"]
        );
        assert_eq!(gate(&model, Kind::CrossRankAggregated)["observations"], 0);
    }
}

#[test]
fn mixed_selector_prior_support_miss_uses_online_without_paired_evidence() {
    let (_fixture, _a, _table, mut model) = selector();
    let mut rows = group(Kind::CrossRankAggregated);
    rows[1].scheduled_requests.num_decode_requests = 8;
    rows[1].scheduled_requests.sum_decode_kv_tokens = 256;
    let before = model.estimate_forward_pass_with_route(&rows).unwrap();
    assert!(before["latency_ms"].is_null());
    assert_eq!(
        before["temporary_mixed_composition"]["stable_available"],
        false
    );
    let c = seed(Kind::CrossRankAggregated, 20);
    model.append_mixed_learned_checkpoint(c.path()).unwrap();
    let after = model.estimate_forward_pass_with_route(&rows).unwrap();
    assert_eq!(after["latency_ms"], 40.0);
    let route = &after["temporary_mixed_composition"];
    assert_eq!(route["preferred_expert"], "stable");
    assert_eq!(route["selected_expert"], "online");
    assert_eq!(route["stable_available"], false);
    assert_eq!(route["online_available"], true);
    assert_eq!(route["route"], "ordinary");
    model.tune_with_fpms(&[rows]).unwrap();
    assert_eq!(
        model.ordinary_accepted_observations(Kind::CrossRankAggregated),
        1
    );
    assert_eq!(
        model.mixed_composition.as_ref().unwrap().admissions[&Kind::CrossRankAggregated],
        0
    );
    assert!(
        model.learned_selector_diagnostics().unwrap()["states"]
            .as_array()
            .unwrap()
            .iter()
            .all(|state| state["observations"] == 0)
    );
}

#[test]
fn mixed_selector_hard_prior_error_keeps_online_admission_visible() {
    let (_fixture, _a, _table, mut model) = selector();
    let c = seed(Kind::ContainsLocallyMixed, 20);
    model.append_mixed_learned_checkpoint(c.path()).unwrap();
    let path = model
        .mixed_composition
        .as_ref()
        .unwrap()
        .engine
        .as_ref()
        .unwrap()
        .database()
        .fpm_forward
        .parquet_path();
    std::fs::write(path, b"corrupted prior").unwrap();
    let rows = group(Kind::ContainsLocallyMixed);
    assert!(model.estimate_forward_pass_time_ms(&rows).is_err());
    assert!(model.tune_with_fpms(&[rows]).is_err());
    assert_eq!(
        model.ordinary_accepted_observations(Kind::ContainsLocallyMixed),
        1
    );
    assert_eq!(
        model.mixed_composition.as_ref().unwrap().admissions[&Kind::ContainsLocallyMixed],
        0
    );
    assert_eq!(gate(&model, Kind::ContainsLocallyMixed)["observations"], 0);
}

#[test]
fn mixed_selector_missing_request_features_counts_rejection_without_admission() {
    fn request_checkpoint(
        kinds: &[Kind],
        count: usize,
    ) -> (tempfile::TempDir, ForwardPassPerfModelConfig) {
        let (dir, config) =
            checkpoint_with_stores(ForwardPassWorkerType::Aggregated, kinds, count, 40.0);
        // The constant forest remains exactly fitted when its one unused input
        // is a request feature. Keep the authenticated checkpoint consistent.
        let artifact_path = dir.path().join("model.json");
        let mut artifact: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&artifact_path).unwrap()).unwrap();
        artifact["features"] = serde_json::json!(["req_batch_size"]);
        let artifact_bytes = serde_json::to_vec(&artifact).unwrap();
        std::fs::write(&artifact_path, &artifact_bytes).unwrap();
        let manifest_path = dir.path().join("manifest.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        manifest["features"] = artifact["features"].clone();
        manifest["artifact_sha256"] = digest(&artifact_bytes).into();
        std::fs::write(manifest_path, manifest.to_string()).unwrap();
        (dir, config)
    }

    let fixture = Fixture::new(0, 0, true);
    let (_a, config) = request_checkpoint(&[Kind::PureDecode, Kind::PurePrefill], 2);
    let (c, _) = request_checkpoint(&[Kind::ContainsLocallyMixed], 20);
    let table = tempfile::tempdir().unwrap();
    let mut model = ForwardPassPerfModel::best_available(config).unwrap();
    model.mixed_composition = Some(fixture.runtime(table.path()));
    model.append_mixed_learned_checkpoint(c.path()).unwrap();
    let before = (
        model.learned_expert_artifacts().unwrap(),
        model.learned_selector_diagnostics(),
        model.mixed_composition_state().unwrap(),
    );
    let mut expected_online = model.learned_online_diagnostics().unwrap();
    assert_eq!(expected_online["rejected_observations"], 0);
    expected_online["rejected_observations"] = 1.into();
    // Aggregate-only telemetry is valid for the interpolation prior, but cannot
    // supply a request-feature forest. Reject once before either history learns.
    let error = model
        .tune_with_fpms(&[group(Kind::ContainsLocallyMixed)])
        .unwrap_err();
    assert!(error.to_string().contains("per-request features"));
    assert_eq!(model.learned_online_diagnostics().unwrap(), expected_online);
    assert_eq!(
        (
            model.learned_expert_artifacts().unwrap(),
            model.learned_selector_diagnostics(),
            model.mixed_composition_state().unwrap()
        ),
        before
    );
}

#[test]
fn mixed_selector_invalid_and_duplicate_append_are_atomic() {
    let (_fixture, _a, _table, mut model) = selector();
    let pure = seed(Kind::PureDecode, 20);
    let short = seed(Kind::ContainsLocallyMixed, 19);
    let (multi, _) = checkpoint_with_stores(
        ForwardPassWorkerType::Aggregated,
        &[Kind::ContainsLocallyMixed, Kind::CrossRankAggregated],
        20,
        40.0,
    );
    let wrong_identity = seed(Kind::ContainsLocallyMixed, 20);
    let manifest_path = wrong_identity.path().join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    manifest["identity"]["model"] = "other-model".into();
    std::fs::write(&manifest_path, manifest.to_string()).unwrap();
    let corrupt = seed(Kind::ContainsLocallyMixed, 20);
    std::fs::write(corrupt.path().join("rows.parquet"), b"corrupt rows").unwrap();
    for path in [
        pure.path(),
        short.path(),
        multi.path(),
        wrong_identity.path(),
        corrupt.path(),
    ] {
        let before = (
            model.learned_expert_artifacts().unwrap(),
            model.learned_online_diagnostics(),
            model.learned_selector_diagnostics(),
            model.mixed_composition_state().unwrap(),
        );
        assert!(model.append_mixed_learned_checkpoint(path).is_err());
        assert_eq!(
            (
                model.learned_expert_artifacts().unwrap(),
                model.learned_online_diagnostics(),
                model.learned_selector_diagnostics(),
                model.mixed_composition_state().unwrap()
            ),
            before
        );
    }
    let valid = seed(Kind::ContainsLocallyMixed, 20);
    model.append_mixed_learned_checkpoint(valid.path()).unwrap();
    let before = (
        model.learned_expert_artifacts().unwrap(),
        model.learned_online_diagnostics(),
        model.learned_selector_diagnostics(),
        model.mixed_composition_state().unwrap(),
    );
    assert!(model.append_mixed_learned_checkpoint(valid.path()).is_err());
    assert_eq!(
        (
            model.learned_expert_artifacts().unwrap(),
            model.learned_online_diagnostics(),
            model.learned_selector_diagnostics(),
            model.mixed_composition_state().unwrap()
        ),
        before
    );
}
