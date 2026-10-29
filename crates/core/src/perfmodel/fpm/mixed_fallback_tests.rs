// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::ScheduledRequestMetrics;
use crate::perfmodel::fpm::estimator::{
    MixedCompositionCalibrationConfig, OriginalASource, TemporaryMixedCompositionScenario,
};

fn group(kind: Kind) -> Vec<ForwardPassMetrics> {
    let prefill = ScheduledRequestMetrics {
        num_prefill_requests: 1,
        sum_prefill_tokens: 100,
        ..Default::default()
    };
    let decode = ScheduledRequestMetrics {
        num_decode_requests: 12,
        sum_decode_kv_tokens: 96,
        ..Default::default()
    };
    let schedules = match kind {
        Kind::PurePrefill => [prefill.clone(), prefill],
        Kind::PureDecode => [decode.clone(), decode],
        Kind::CrossRankAggregated => [prefill, decode],
        Kind::ContainsLocallyMixed => [
            ScheduledRequestMetrics {
                num_prefill_requests: 1,
                sum_prefill_tokens: 100,
                ..decode.clone()
            },
            ScheduledRequestMetrics {
                num_decode_requests: 8,
                sum_decode_kv_tokens: 64,
                ..Default::default()
            },
        ],
    };
    schedules
        .into_iter()
        .enumerate()
        .map(|(rank, scheduled_requests)| ForwardPassMetrics {
            dp_rank: rank as u32,
            // Telemetry is seconds; predictions and ratio denominators are ms.
            wall_time: if rank == 0 { 0.016 } else { 0.0 },
            scheduled_requests,
            ..Default::default()
        })
        .collect()
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

struct Fixture {
    _dir: tempfile::TempDir,
    config: ForwardPassPerfModelConfig,
    learned: LearnedForwardPassModel,
    manifest: serde_json::Value,
}

impl Fixture {
    fn new(cross_rows: usize, local_rows: usize, calibrated: bool) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let mut config = ForwardPassPerfModelConfig::new(
            "test/model",
            "test-gpu",
            crate::BackendKind::Vllm,
            ForwardPassWorkerType::Aggregated,
        );
        config.attention_dp = 2;
        config.estimation_mode = EstimationMode::FpmLearned;
        config.estimator_config.fpm_learned.adaptation =
            crate::perfmodel::fpm::LearnedAdaptation::Frozen;
        let identity = canonical_learned_identity(&config).unwrap();
        let counts = [
            (Kind::PureDecode, 2),
            (Kind::PurePrefill, 2),
            (Kind::CrossRankAggregated, cross_rows),
            (Kind::ContainsLocallyMixed, local_rows),
        ];
        let mut stores = serde_json::Map::new();
        let mut train_rows = serde_json::Map::new();
        let mut groups = Vec::new();
        for (kind, count) in counts {
            for _ in 0..count {
                groups.push(group(kind));
            }
            if count >= 2 {
                let key = serde_json::to_value(kind)
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_owned();
                stores.insert(
                    key.clone(),
                    serde_json::json!({"baseline": 99.0, "trees": []}),
                );
                train_rows.insert(key, serde_json::json!(count));
            }
        }
        let artifact = serde_json::json!({
            "schema": "aic_fpm_learned_forward_perf", "schema_version": 1,
            "worker_type": "aggregated", "target": "ms",
            "features": ["num_decode_requests", "sum_decode_kv_tokens"],
            "stores": stores, "metadata": {"identity": identity, "train_rows": train_rows}
        });
        let seed = serde_json::to_vec(&artifact).unwrap();
        std::fs::write(dir.path().join("seed.json"), &seed).unwrap();
        let source = serde_json::to_vec(&groups).unwrap();
        std::fs::write(dir.path().join("source.json"), &source).unwrap();
        let manifest = serde_json::json!({
            "schema_version": 1, "source_kind": "self_benchmark", "identity": identity,
            "min_store_rows": 2, "accepted_iterations": groups,
            "seed_artifacts": [{"path": "seed.json", "sha256": digest(&seed)}],
            "provenance": {"source_sha256": digest(&source),
                "grouping": {"mode": "observed", "attention_dp": 2},
                "sources": [{"path": "source.json", "sha256": digest(&source)}]}
        });
        config.estimator_config.temporary_mixed_composition =
            Some(TemporaryMixedCompositionConfig {
                scenario: if calibrated {
                    TemporaryMixedCompositionScenario::AC
                } else {
                    TemporaryMixedCompositionScenario::A
                },
                mode: if calibrated {
                    TemporaryMixedCompositionMode::ComposeCalibrated
                } else {
                    TemporaryMixedCompositionMode::Compose
                },
                original_a: OriginalASource {
                    manifest_path: dir.path().join("manifest.json"),
                    source_sha256: None,
                },
                calibration: MixedCompositionCalibrationConfig::default(),
                quality_gate: None,
            });
        let learned = LearnedForwardPassModel::from_json(&artifact.to_string()).unwrap();
        let mut fixture = Self {
            _dir: dir,
            config,
            learned,
            manifest,
        };
        fixture.write_manifest();
        fixture
    }

    fn write_manifest(&mut self) {
        let bytes = serde_json::to_vec(&self.manifest).unwrap();
        let policy = self
            .config
            .estimator_config
            .temporary_mixed_composition
            .as_mut()
            .unwrap();
        std::fs::write(&policy.original_a.manifest_path, &bytes).unwrap();
        policy.original_a.source_sha256 = Some(digest(&bytes));
    }

    fn analyze(&self) -> MixedComposition {
        MixedComposition::analyze(&self.config, &self.learned)
            .unwrap()
            .unwrap()
    }

    fn runtime(&self, table_dir: &Path) -> MixedComposition {
        let mut mixed = self.analyze();
        let model = crate::perfmodel::fpm::tests::direct_coverage_model(table_dir);
        let ForwardPassPerfMode::Native { engine, .. } = model.mode else {
            panic!("fixture is native")
        };
        mixed.engine = Some(engine);
        mixed.prior = serde_json::json!({"test_fixture": "direct_raw_measurements"});
        if mixed.policy.mode == TemporaryMixedCompositionMode::ComposeCalibrated {
            let axes = mixed
                .bounds
                .map(|value| AxisRange::from_zero_to_f64(value).unwrap());
            for route in &mixed.routes {
                if route.fallback_eligible {
                    mixed.corrections.insert(
                        route.workload_kind,
                        CorrectionBuckets::new_mixed_hybrid(
                            &mixed.policy.calibration.options(),
                            &axes,
                        )
                        .unwrap(),
                    );
                    mixed.admissions.insert(route.workload_kind, 0);
                }
            }
        }
        mixed
    }
}

#[test]
fn mixed_original_a_stream_preserves_decoded_groups_and_checks_every_byte() {
    let mut fixture = Fixture::new(2, 2, false);
    // Cross multiple read-buffer boundaries, including metadata after groups.
    fixture.manifest["provenance"]["notes"] = serde_json::json!("x".repeat(65536));
    fixture.write_manifest();
    let source = &fixture
        .config
        .estimator_config
        .temporary_mixed_composition
        .as_ref()
        .unwrap()
        .original_a;
    let bytes = std::fs::read(&source.manifest_path).unwrap();
    let reference: OriginalAManifest = serde_json::from_slice(&bytes).unwrap();
    let streamed = checked_manifest(&source.manifest_path, &digest(&bytes)).unwrap();
    assert_eq!(streamed.accepted_iterations, reference.accepted_iterations);
    assert_eq!(streamed.identity, reference.identity);
    assert_eq!(streamed.provenance, reference.provenance);

    let mut malformed = vec![b'!'];
    malformed.extend_from_slice(&bytes);
    std::fs::write(&source.manifest_path, &malformed).unwrap();
    // An early parse error must not stop digest validation of the whole input.
    assert!(
        checked_manifest(&source.manifest_path, &digest(&bytes))
            .err()
            .unwrap()
            .to_string()
            .contains("source digest mismatch")
    );
    let parse_error = checked_manifest(&source.manifest_path, &digest(&malformed))
        .err()
        .unwrap()
        .to_string();
    assert!(parse_error.contains("expected value"), "{parse_error}");
    assert!(!parse_error.contains("source digest mismatch"));
}

#[test]
fn mixed_original_a_routes_cover_absent_partial_and_complete_support() {
    for (cross, local, expected) in [
        (0, 0, [true, true]),
        (2, 1, [false, true]),
        (1, 2, [true, false]),
        (2, 2, [false, false]),
    ] {
        let fixture = Fixture::new(cross, local, true);
        let mixed = fixture.analyze();
        for (kind, owned) in MIXED_KINDS.into_iter().zip(expected) {
            assert_eq!(mixed.owned_feature(&group(kind)).unwrap().is_some(), owned);
            let route = mixed.route(&group(kind), false).unwrap();
            assert_eq!(route["owned"], owned);
            assert_eq!(route["calibration_enabled"], owned);
        }
        // Pure paths retain the existing acceptance contract even if callers
        // omit idle ranks; the new completeness requirement is mixed-only.
        let pure = group(Kind::PureDecode);
        assert!(mixed.owned_feature(&pure[..1]).unwrap().is_none());
        assert!(
            mixed
                .owned_feature(&group(Kind::PurePrefill))
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn mixed_original_a_rejects_corrupt_inconsistent_and_missing_sources() {
    let mut fixture = Fixture::new(0, 0, true);
    fixture.manifest["accepted_iterations"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::to_value(group(Kind::PureDecode)).unwrap());
    fixture.write_manifest();
    assert!(
        MixedComposition::analyze(&fixture.config, &fixture.learned)
            .unwrap_err()
            .to_string()
            .contains("group count")
    );

    let mut fixture = Fixture::new(0, 0, true);
    fixture.manifest["accepted_iterations"][0][1]["dp_rank"] = serde_json::json!(0);
    fixture.write_manifest();
    assert!(MixedComposition::analyze(&fixture.config, &fixture.learned).is_err());

    let fixture = Fixture::new(0, 0, true);
    std::fs::write(fixture._dir.path().join("source.json"), b"changed source").unwrap();
    assert!(
        MixedComposition::analyze(&fixture.config, &fixture.learned)
            .unwrap_err()
            .to_string()
            .contains("digest mismatch")
    );

    let fixture = Fixture::new(0, 0, true);
    std::fs::remove_file(fixture._dir.path().join("seed.json")).unwrap();
    assert!(matches!(
        MixedComposition::analyze(&fixture.config, &fixture.learned),
        Err(AicError::Io { .. })
    ));

    let fixture = Fixture::new(2, 0, true);
    let mut artifact = fixture.learned.artifact();
    artifact.stores.remove(&Kind::CrossRankAggregated);
    let current = LearnedForwardPassModel::from_artifact(artifact).unwrap();
    assert!(
        MixedComposition::analyze(&fixture.config, &current)
            .unwrap_err()
            .to_string()
            .contains("original A")
    );
}

#[test]
fn mixed_original_a_verifies_frozen_stores_and_online_checkpoint_anchor() {
    let mut fixture = Fixture::new(0, 0, true);
    let mut altered = fixture.learned.artifact();
    altered.stores.get_mut(&Kind::PureDecode).unwrap().baseline += 1.0;
    let altered = LearnedForwardPassModel::from_artifact(altered).unwrap();
    assert!(MixedComposition::analyze(&fixture.config, &altered).is_err());
    assert!(
        fixture
            .analyze()
            .verify_online_anchor(&"a".repeat(64))
            .is_err()
    );

    let checkpoint_dir = fixture._dir.path().join("checkpoint");
    std::fs::create_dir(&checkpoint_dir).unwrap();
    let seed = std::fs::read(fixture._dir.path().join("seed.json")).unwrap();
    std::fs::write(checkpoint_dir.join("artifact.json"), &seed).unwrap();
    let mut checkpoint = serde_json::json!({
        "schema": "aisimulate_fpm_training_checkpoint",
        "artifact_sha256": digest(&seed),
        "identity": canonical_learned_identity(&fixture.config).unwrap()
    });
    let bytes = serde_json::to_vec(&checkpoint).unwrap();
    std::fs::write(checkpoint_dir.join("manifest.json"), &bytes).unwrap();
    let anchor = digest(&bytes);
    fixture.manifest["seed_artifacts"][0]["path"] = serde_json::json!("checkpoint/artifact.json");
    fixture.manifest["seed_artifacts"][0]["checkpoint_manifest_sha256"] = serde_json::json!(anchor);
    fixture.write_manifest();
    let mixed = fixture.analyze();
    mixed.verify_online_anchor(&anchor).unwrap();
    assert!(mixed.verify_online_anchor(&"a".repeat(64)).is_err());

    checkpoint["artifact_sha256"] = serde_json::json!("b".repeat(64));
    let bytes = serde_json::to_vec(&checkpoint).unwrap();
    std::fs::write(checkpoint_dir.join("manifest.json"), &bytes).unwrap();
    fixture.manifest["seed_artifacts"][0]["checkpoint_manifest_sha256"] =
        serde_json::json!(digest(&bytes));
    fixture.write_manifest();
    assert!(
        MixedComposition::analyze(&fixture.config, &fixture.learned)
            .unwrap_err()
            .to_string()
            .contains("does not bind the seed artifact")
    );
}

#[test]
fn mixed_calibration_is_class_local_causal_and_relative_to_raw_prior() {
    let fixture = Fixture::new(0, 0, true);
    let table = tempfile::tempdir().unwrap();
    let mut mixed = fixture.runtime(table.path());
    let mut cross = group(Kind::CrossRankAggregated);
    let mut local = group(Kind::ContainsLocallyMixed);
    let cross_feature = mixed.owned_feature(&cross).unwrap().unwrap();
    let local_feature = mixed.owned_feature(&local).unwrap().unwrap();
    // Literal direct measurements: cross=max(prefill10,decode16), local=
    // max(prefill10+decode16-baseline12,other_rank_decode8).
    assert_eq!(
        mixed
            .predict(&cross, &cross_feature, false)
            .unwrap()
            .latency_ms,
        Some(16.0)
    );
    assert_eq!(
        mixed
            .predict(&local, &local_feature, false)
            .unwrap()
            .latency_ms,
        Some(14.0)
    );
    cross[0].wall_time = 0.0176;
    for _ in 0..4 {
        mixed.observe(&cross, &cross_feature).unwrap();
    }
    assert_eq!(
        mixed
            .predict(&cross, &cross_feature, false)
            .unwrap()
            .latency_ms,
        Some(16.0)
    );
    mixed.observe(&cross, &cross_feature).unwrap();
    assert_eq!(
        mixed
            .predict(&cross, &cross_feature, false)
            .unwrap()
            .latency_ms,
        Some(17.6)
    );
    assert_eq!(
        mixed
            .predict(&local, &local_feature, false)
            .unwrap()
            .latency_ms,
        Some(14.0)
    );
    local[0].wall_time = 0.028;
    for _ in 0..5 {
        mixed.observe(&local, &local_feature).unwrap();
    }
    assert_eq!(
        mixed
            .predict(&local, &local_feature, false)
            .unwrap()
            .latency_ms,
        Some(28.0)
    );
    assert_eq!(
        mixed
            .predict(&cross, &cross_feature, false)
            .unwrap()
            .latency_ms,
        Some(17.6)
    );
    // Later labels use y/raw, never y/the already corrected answer.
    cross[0].wall_time = 0.018;
    for _ in 0..6 {
        mixed.observe(&cross, &cross_feature).unwrap();
    }
    assert_eq!(
        mixed
            .predict(&cross, &cross_feature, false)
            .unwrap()
            .latency_ms,
        Some(18.0)
    );
    assert_eq!(mixed.admissions[&Kind::CrossRankAggregated], 11);
    assert_eq!(mixed.admissions[&Kind::ContainsLocallyMixed], 5);
}

#[test]
fn mixed_missing_ordinary_store_uses_fallback_and_preserves_pure_paths() {
    let fixture = Fixture::new(0, 0, true);
    let table = tempfile::tempdir().unwrap();
    let mut model = ForwardPassPerfModel::from_learned(
        &fixture.learned.to_json().unwrap(),
        ForwardPassPerfOptions::default(),
    )
    .unwrap();
    model.mixed_composition = Some(fixture.runtime(table.path()));
    let mut cross = group(Kind::CrossRankAggregated);
    cross[0].wall_time = 0.0176;
    assert_eq!(
        model.estimate_forward_pass_time_ms(&cross).unwrap(),
        Some(16.0)
    );
    model.tune_with_fpms(&vec![cross.clone(); 5]).unwrap();
    assert_eq!(
        model.estimate_forward_pass_time_ms(&cross).unwrap(),
        Some(17.6)
    );
    let ForwardPassPerfMode::Learned {
        corrections,
        learned,
        ..
    } = &model.mode
    else {
        panic!("learned model")
    };
    assert_eq!(corrections.observation_count(), 0);
    assert!(!learned.has_store(Kind::CrossRankAggregated));
    let mixed_state = model
        .mixed_composition
        .as_ref()
        .unwrap()
        .snapshot()
        .unwrap();
    let mut pure = group(Kind::PureDecode);
    assert_eq!(
        model.estimate_forward_pass_time_ms(&pure).unwrap(),
        Some(99.0)
    );
    pure[0].wall_time = 0.198;
    model.tune_with_fpms(&vec![pure.clone(); 5]).unwrap();
    assert_eq!(
        model.estimate_forward_pass_time_ms(&pure).unwrap(),
        Some(198.0)
    );
    assert_eq!(
        model
            .mixed_composition
            .as_ref()
            .unwrap()
            .snapshot()
            .unwrap(),
        mixed_state
    );
    assert_eq!(
        model.estimate_forward_pass_time_ms(&cross).unwrap(),
        Some(17.6)
    );
}

#[test]
fn mixed_a_composition_remains_raw_and_unowned_stores_do_not_change_routes() {
    let mut fixture = Fixture::new(0, 0, false);
    let mut artifact = fixture.learned.artifact();
    let pure = artifact.stores[&Kind::PureDecode].clone();
    artifact
        .stores
        .insert(Kind::CrossRankAggregated, pure.clone());
    artifact.stores.insert(Kind::ContainsLocallyMixed, pure);
    fixture.learned = LearnedForwardPassModel::from_artifact(artifact).unwrap();
    let table = tempfile::tempdir().unwrap();
    let mut mixed = fixture.runtime(table.path());
    let mut query = group(Kind::ContainsLocallyMixed);
    query[0].wall_time = 1000.0;
    let feature = mixed.owned_feature(&query).unwrap().unwrap();
    for _ in 0..10 {
        mixed.observe(&query, &feature).unwrap();
    }
    let estimate = mixed.predict(&query, &feature, true).unwrap();
    assert_eq!(estimate.native_latency_ms, Some(14.0));
    assert_eq!(estimate.latency_ms, Some(14.0));
    assert_eq!(estimate.correction_factor, Some(1.0));
    assert!(mixed.corrections.is_empty());
    assert!(mixed.admissions.is_empty());
}

#[test]
fn mixed_support_misses_do_not_admit_and_corrupt_data_remains_an_error() {
    let fixture = Fixture::new(0, 0, true);
    let table = tempfile::tempdir().unwrap();
    let mut mixed = fixture.runtime(table.path());
    let mut query = group(Kind::CrossRankAggregated);
    query[1].scheduled_requests.num_decode_requests = 8;
    query[1].scheduled_requests.sum_decode_kv_tokens = 256;
    let feature = mixed.owned_feature(&query).unwrap().unwrap();
    let estimate = mixed.predict(&query, &feature, false).unwrap();
    assert_eq!(estimate.latency_ms, None);
    assert!(estimate.unavailable_reason.is_some());
    mixed.observe(&query, &feature).unwrap();
    assert_eq!(mixed.admissions[&Kind::CrossRankAggregated], 0);

    let table = tempfile::tempdir().unwrap();
    let corrupted = fixture.runtime(table.path());
    let path = corrupted
        .engine
        .as_ref()
        .unwrap()
        .database()
        .fpm_forward
        .parquet_path();
    std::fs::write(path, b"not parquet").unwrap();
    let query = group(Kind::CrossRankAggregated);
    let feature = corrupted.owned_feature(&query).unwrap().unwrap();
    let error = corrupted.predict(&query, &feature, false).unwrap_err();
    assert!(!matches!(error, AicError::FpmQueryUnsupported(_)));
}

#[test]
fn mixed_invalid_feedback_cannot_advance_or_change_calibration() {
    let fixture = Fixture::new(0, 0, true);
    let table = tempfile::tempdir().unwrap();
    let mut mixed = fixture.runtime(table.path());
    let before = mixed.snapshot().unwrap();
    for invalid_time in [f64::NAN, f64::INFINITY, -1.0] {
        let mut query = group(Kind::CrossRankAggregated);
        query[1].wall_time = invalid_time;
        let feature = mixed.owned_feature(&query).unwrap().unwrap();
        assert!(mixed.observe(&query, &feature).is_err());
        assert_eq!(mixed.snapshot().unwrap(), before);
    }
    let local = group(Kind::ContainsLocallyMixed);
    assert!(mixed.owned_feature(&local[..1]).is_err());
    let mut duplicated = local.clone();
    duplicated[1].dp_rank = duplicated[0].dp_rank;
    assert!(mixed.owned_feature(&duplicated).is_err());
    assert_eq!(mixed.snapshot().unwrap(), before);
}

#[test]
fn mixed_rank_permutation_preserves_scores_latency_and_calibration_state() {
    let fixture = Fixture::new(0, 0, true);
    let table = tempfile::tempdir().unwrap();
    let mixed = fixture.runtime(table.path());
    for kind in MIXED_KINDS {
        let ordered = group(kind);
        let mut permuted = ordered.clone();
        permuted.reverse();
        let feature = mixed.owned_feature(&ordered).unwrap().unwrap();
        assert_eq!(mixed.owned_feature(&permuted).unwrap().unwrap(), feature);
        let first = mixed.predict(&ordered, &feature, true).unwrap();
        let second = mixed.predict(&permuted, &feature, true).unwrap();
        assert_eq!(first.latency_ms, second.latency_ms);
        assert_eq!(first.native_latency_ms, second.native_latency_ms);
        assert_eq!(first.correction_factor, second.correction_factor);
        // Evidence retains the Engine contract: indices locate entries in the
        // supplied slice. Mapping back to telemetry IDs identifies the same rank.
        assert_eq!(
            first.max_rank.map(|index| ordered[index].dp_rank),
            second.max_rank.map(|index| permuted[index].dp_rank)
        );
        let by_id = |estimate: crate::ForwardPassEstimate, rows: &[ForwardPassMetrics]| {
            estimate
                .ranks
                .into_iter()
                .map(|rank| (rows[rank.rank].dp_rank, rank.latency_ms))
                .collect::<BTreeMap<_, _>>()
        };
        assert_eq!(by_id(first, &ordered), by_id(second, &permuted));
        let mut first = mixed.clone();
        let mut second = mixed.clone();
        for _ in 0..5 {
            first.observe(&ordered, &feature).unwrap();
            second.observe(&permuted, &feature).unwrap();
        }
        assert_eq!(first.snapshot().unwrap(), second.snapshot().unwrap());
        assert_eq!(
            first.predict(&ordered, &feature, false).unwrap().latency_ms,
            second
                .predict(&permuted, &feature, false)
                .unwrap()
                .latency_ms
        );
    }
}

#[test]
fn mixed_exhausted_admission_counter_fails_before_any_mutation() {
    let fixture = Fixture::new(0, 0, true);
    let table = tempfile::tempdir().unwrap();
    let mut mixed = fixture.runtime(table.path());
    let query = group(Kind::CrossRankAggregated);
    let feature = mixed.owned_feature(&query).unwrap().unwrap();
    mixed
        .admissions
        .insert(feature.workload_kind, usize::MAX as u64);
    let before = mixed.snapshot().unwrap();
    let error = mixed.observe(&query, &feature).unwrap_err();
    assert!(error.to_string().contains("counter exhausted"));
    assert_eq!(mixed.snapshot().unwrap(), before);
}

#[test]
fn mixed_snapshot_binds_prior_routes_and_counter_progress_atomically() {
    let fixture = Fixture::new(0, 0, true);
    let table = tempfile::tempdir().unwrap();
    let mut mixed = fixture.runtime(table.path());
    let query = group(Kind::CrossRankAggregated);
    let feature = mixed.owned_feature(&query).unwrap().unwrap();
    for _ in 0..5 {
        mixed.observe(&query, &feature).unwrap();
    }
    let snapshot = mixed.snapshot().unwrap();
    let restored_table = tempfile::tempdir().unwrap();
    let mut restored = fixture.runtime(restored_table.path());
    restored.restore(snapshot.clone()).unwrap();
    assert_eq!(restored.snapshot().unwrap(), snapshot);
    for pointer in [
        "/version",
        "/binding/prior",
        "/binding/routes/0/fallback_eligible",
        "/admissions/cross_rank_aggregated",
    ] {
        let mut changed = snapshot.clone();
        *changed.pointer_mut(pointer).unwrap() = if pointer.starts_with("/admissions") {
            serde_json::json!(99)
        } else {
            serde_json::json!("changed")
        };
        assert!(mixed.restore(changed).is_err(), "{pointer}");
        assert_eq!(mixed.snapshot().unwrap(), snapshot, "{pointer}");
    }
}

// The ordinary constant forest is an independent 99 ms oracle; interpolation
// is 16 ms cross-rank / 14 ms locally mixed from the direct fixture above.
fn ready_mixed_model(fixture: &Fixture) -> ForwardPassPerfModel {
    let mut artifact = fixture.learned.artifact();
    let store = artifact.stores[&Kind::PureDecode].clone();
    for kind in MIXED_KINDS {
        artifact.stores.insert(kind, store.clone());
    }
    ForwardPassPerfModel::from_learned(
        &LearnedForwardPassModel::from_artifact(artifact)
            .unwrap()
            .to_json()
            .unwrap(),
        ForwardPassPerfOptions::default(),
    )
    .unwrap()
}

#[test]
fn mixed_ac_prefers_ready_ordinary_without_querying_prior_or_its_rank_validator() {
    let fixture = Fixture::new(0, 0, true);
    let table = tempfile::tempdir().unwrap();
    let mut model = ready_mixed_model(&fixture);
    let mixed = fixture.runtime(table.path());
    // A malformed prior would be a hard error if queried. Ordinary success
    // must not evaluate it or require the fallback's complete DP group.
    std::fs::write(
        mixed
            .engine
            .as_ref()
            .unwrap()
            .database()
            .fpm_forward
            .parquet_path(),
        b"invalid",
    )
    .unwrap();
    model.mixed_composition = Some(mixed);
    let query = group(Kind::ContainsLocallyMixed);
    for rows in [&query[..], &query[..1]] {
        assert_eq!(
            model.estimate_forward_pass_time_ms(rows).unwrap(),
            Some(99.0)
        );
        let detailed = model.estimate_forward_pass_detailed(rows).unwrap();
        assert_eq!(detailed.latency_ms, Some(99.0));
        assert_eq!(detailed.native_latency_ms, None);
        assert!(detailed.ranks.is_empty());
        let route = model.mixed_composition_route(rows).unwrap().unwrap();
        assert_eq!(route["route"], "ordinary");
        assert_eq!(route["owned"], false);
        assert_eq!(route["fallback_eligible"], true);
        assert_eq!(route["calibration_enabled"], true);
    }
    let idle = [ForwardPassMetrics::default()];
    assert_eq!(
        model.estimate_forward_pass_time_ms(&idle).unwrap(),
        Some(0.0)
    );
    assert!(model.estimate_forward_pass_time_ms(&[]).is_err());
    assert_eq!(
        model
            .estimate_forward_pass_detailed(&idle)
            .unwrap()
            .latency_ms,
        Some(0.0)
    );
    let mut invalid = query;
    invalid[0].scheduled_requests.num_prefill_requests = 0;
    assert!(model.estimate_forward_pass_time_ms(&invalid).is_err());
    assert!(model.estimate_forward_pass_detailed(&invalid).is_err());
}

#[test]
fn mixed_ac_preserves_ordinary_updates_and_independent_fallback_history() {
    let fixture = Fixture::new(0, 0, true);
    let table = tempfile::tempdir().unwrap();
    let mut enabled = ready_mixed_model(&fixture);
    let mut ordinary = ready_mixed_model(&fixture);
    enabled.mixed_composition = Some(fixture.runtime(table.path()));
    let mut query = group(Kind::ContainsLocallyMixed);
    query[0].wall_time = 0.028;
    for count in 1..=6 {
        assert_eq!(
            enabled.estimate_forward_pass_time_ms(&query).unwrap(),
            ordinary.estimate_forward_pass_time_ms(&query).unwrap()
        );
        enabled.tune_with_fpms(&[query.clone()]).unwrap();
        ordinary.tune_with_fpms(&[query.clone()]).unwrap();
        assert_eq!(
            enabled.mixed_composition.as_ref().unwrap().admissions[&Kind::ContainsLocallyMixed],
            count
        );
        assert_eq!(
            enabled.estimate_forward_pass_time_ms(&query).unwrap(),
            ordinary.estimate_forward_pass_time_ms(&query).unwrap()
        );
    }
    let mixed = enabled.mixed_composition.as_ref().unwrap();
    let feature = mixed.owned_feature(&query).unwrap().unwrap();
    assert_eq!(
        mixed.predict(&query, &feature, false).unwrap().latency_ms,
        Some(28.0)
    );
    let state = mixed.snapshot().unwrap();
    let other_table = tempfile::tempdir().unwrap();
    let mut resumed = fixture.runtime(other_table.path());
    resumed.restore(state.clone()).unwrap();
    assert_eq!(resumed.snapshot().unwrap(), state);
    let mut legacy = state.clone();
    legacy["version"] = serde_json::json!("original_a_composition_attention_moe_v1");
    assert!(resumed.restore(legacy).is_err());
    assert_eq!(resumed.snapshot().unwrap(), state);
}

#[test]
fn mixed_ac_support_miss_does_not_suppress_ordinary_feedback() {
    let fixture = Fixture::new(0, 0, true);
    let table = tempfile::tempdir().unwrap();
    let mut enabled = ready_mixed_model(&fixture);
    let mut ordinary = ready_mixed_model(&fixture);
    enabled.mixed_composition = Some(fixture.runtime(table.path()));
    let mut query = group(Kind::CrossRankAggregated);
    query[1].scheduled_requests.num_decode_requests = 8;
    query[1].scheduled_requests.sum_decode_kv_tokens = 256;
    query[0].wall_time = 0.198;
    for _ in 0..5 {
        enabled.tune_with_fpms(&[query.clone()]).unwrap();
        ordinary.tune_with_fpms(&[query.clone()]).unwrap();
    }
    assert_eq!(
        enabled.estimate_forward_pass_time_ms(&query).unwrap(),
        Some(198.0)
    );
    assert_eq!(
        enabled.estimate_forward_pass_time_ms(&query).unwrap(),
        ordinary.estimate_forward_pass_time_ms(&query).unwrap()
    );
    assert_eq!(
        enabled.mixed_composition.as_ref().unwrap().admissions[&Kind::CrossRankAggregated],
        0
    );
}

#[test]
fn mixed_a_public_path_keeps_composition_priority_and_does_not_learn() {
    let fixture = Fixture::new(0, 0, false);
    let table = tempfile::tempdir().unwrap();
    let mut model = ready_mixed_model(&fixture);
    model.mixed_composition = Some(fixture.runtime(table.path()));
    let query = group(Kind::ContainsLocallyMixed);
    assert_eq!(
        model.estimate_forward_pass_time_ms(&query).unwrap(),
        Some(14.0)
    );
    model.tune_with_fpms(&vec![query.clone(); 5]).unwrap();
    assert_eq!(
        model.estimate_forward_pass_time_ms(&query).unwrap(),
        Some(14.0)
    );
    assert_eq!(model.estimate_ordinary_time_ms(&query).unwrap(), Some(99.0));
    let route = model.mixed_composition_route(&query).unwrap().unwrap();
    assert_eq!(route["owned"], true);
    assert_eq!(route["fallback_eligible"], true);
    assert_eq!(route["calibration_enabled"], false);
}

#[test]
fn mixed_ac_preserves_ready_online_trajectory() {
    // Attach the independently validated raw-prior fixture after constructing
    // the ordinary model through a real checkpoint. This isolates dispatcher
    // semantics from source construction, which has separate manifest tests.
    let fixture = Fixture::new(0, 0, true);
    for adaptation in [crate::perfmodel::fpm::LearnedAdaptation::Online] {
        let (_dir, mut config) =
            super::super::combined::tests::checkpoint(ForwardPassWorkerType::Aggregated);
        config.estimator_config.fpm_learned.adaptation = adaptation;
        if adaptation == crate::perfmodel::fpm::LearnedAdaptation::Online {
            config.estimator_config.correction.enabled = false;
            config.estimator_config.selector = Default::default();
        }
        let mut enabled = ForwardPassPerfModel::best_available(config.clone()).unwrap();
        let mut ordinary = ForwardPassPerfModel::best_available(config).unwrap();
        let table = tempfile::tempdir().unwrap();
        enabled.mixed_composition = Some(fixture.runtime(table.path()));
        let mut query = group(Kind::ContainsLocallyMixed);
        for (index, target) in [12.0, 24.0, 14.0, 28.0, 17.0, 30.0, 22.0]
            .into_iter()
            .enumerate()
        {
            query[0].wall_time = target / 1000.0;
            assert_eq!(
                enabled.estimate_forward_pass_time_ms(&query).unwrap(),
                ordinary.estimate_forward_pass_time_ms(&query).unwrap()
            );
            assert_eq!(
                enabled
                    .estimate_forward_pass_detailed(&query)
                    .unwrap()
                    .latency_ms,
                ordinary.estimate_forward_pass_time_ms(&query).unwrap()
            );
            enabled.tune_with_fpms(&[query.clone()]).unwrap();
            ordinary.tune_with_fpms(&[query.clone()]).unwrap();
            assert_eq!(
                enabled.learned_online_diagnostics(),
                ordinary.learned_online_diagnostics()
            );
            assert_eq!(
                enabled.learned_selector_diagnostics(),
                ordinary.learned_selector_diagnostics()
            );
            assert_eq!(
                enabled.mixed_composition.as_ref().unwrap().admissions[&Kind::ContainsLocallyMixed],
                index as u64 + 1
            );
        }
    }
}

#[test]
fn mixed_ac_failed_fit_commits_both_once_and_blocked_retry_does_not_admit() {
    let fixture = Fixture::new(0, 0, true);
    let (_dir, mut config) =
        super::super::combined::tests::checkpoint(ForwardPassWorkerType::Aggregated);
    config.estimator_config.fpm_learned.adaptation =
        crate::perfmodel::fpm::LearnedAdaptation::Online;
    config.estimator_config.fpm_learned.update_every = 2;
    config.estimator_config.correction.enabled = false;
    config.estimator_config.selector = Default::default();
    let mut model = ForwardPassPerfModel::best_available(config).unwrap();
    let table = tempfile::tempdir().unwrap();
    model.mixed_composition = Some(fixture.runtime(table.path()));
    let mut query = group(Kind::ContainsLocallyMixed);
    // Two finite 1e308 ms labels overflow the fit's aggregate, after admission.
    query[0].wall_time = 1e305;
    model.tune_with_fpms(&[query.clone()]).unwrap();
    assert!(model.tune_with_fpms(&[query.clone()]).is_err());
    assert_eq!(
        model.ordinary_accepted_observations(Kind::ContainsLocallyMixed),
        2
    );
    assert_eq!(
        model.mixed_composition.as_ref().unwrap().admissions[&Kind::ContainsLocallyMixed],
        2
    );
    let state = model.mixed_composition_state().unwrap();
    assert!(
        model
            .tune_with_fpms(&[query])
            .unwrap_err()
            .to_string()
            .contains("failed batch is pending")
    );
    assert_eq!(
        model.ordinary_accepted_observations(Kind::ContainsLocallyMixed),
        2
    );
    assert_eq!(model.mixed_composition_state().unwrap(), state);
    assert!(model.flush_updates().is_err());
    assert_eq!(model.mixed_composition_state().unwrap(), state);
}

#[test]
fn mixed_ac_hard_prior_error_preserves_ordinary_admission_without_mixed_commit() {
    let fixture = Fixture::new(0, 0, true);
    let (_dir, mut config) =
        super::super::combined::tests::checkpoint(ForwardPassWorkerType::Aggregated);
    config.estimator_config.fpm_learned.adaptation =
        crate::perfmodel::fpm::LearnedAdaptation::Online;
    config.estimator_config.correction.enabled = false;
    config.estimator_config.selector = Default::default();
    let mut enabled = ForwardPassPerfModel::best_available(config.clone()).unwrap();
    let mut ordinary = ForwardPassPerfModel::best_available(config).unwrap();
    let table = tempfile::tempdir().unwrap();
    let mixed = fixture.runtime(table.path());
    std::fs::write(
        mixed
            .engine
            .as_ref()
            .unwrap()
            .database()
            .fpm_forward
            .parquet_path(),
        b"invalid",
    )
    .unwrap();
    enabled.mixed_composition = Some(mixed);
    let state = enabled.mixed_composition_state().unwrap();
    let mut query = group(Kind::ContainsLocallyMixed);
    for (index, target) in [20.0, 28.0, 14.0].into_iter().enumerate() {
        query[0].wall_time = target / 1000.0;
        let error = enabled.tune_with_fpms(&[query.clone()]).unwrap_err();
        assert!(!matches!(
            error,
            AicError::FpmQueryUnsupported(_) | AicError::InterpolationCoverage(_)
        ));
        ordinary.tune_with_fpms(&[query.clone()]).unwrap();
        assert_eq!(
            enabled.ordinary_accepted_observations(Kind::ContainsLocallyMixed),
            index as u64 + 1
        );
        assert_eq!(
            enabled.export_learned_artifact().unwrap(),
            ordinary.export_learned_artifact().unwrap()
        );
        assert_eq!(
            enabled.learned_online_diagnostics(),
            ordinary.learned_online_diagnostics()
        );
        assert_eq!(enabled.mixed_composition_state().unwrap(), state);
    }
}

#[path = "mixed_selector_tests.rs"]
mod mixed_selector_tests;

#[test]
fn mixed_quality_gate_checks_future_labels_for_online() {
    use crate::perfmodel::fpm::model::combined::tests::checkpoint_with_stores;
    use crate::perfmodel::fpm::{LearnedAdaptation, MixedQualityGateConfig};
    let mut fixture = Fixture::new(0, 0, true);
    fixture
        .config
        .estimator_config
        .temporary_mixed_composition
        .as_mut()
        .unwrap()
        .quality_gate = Some(MixedQualityGateConfig {
        alpha: 1.0,
        min_comparisons: 1,
    });
    let (_a, mut config) = checkpoint_with_stores(
        ForwardPassWorkerType::Aggregated,
        &[Kind::PureDecode, Kind::PurePrefill],
        20,
        10.0,
    );
    config.estimator_config.fpm_learned.adaptation = LearnedAdaptation::Online;
    config.estimator_config.fpm_learned.update_every = 1000;
    config.estimator_config.correction.enabled = false;
    config.estimator_config.selector = Default::default();
    let table = tempfile::tempdir().unwrap();
    let mut model = ForwardPassPerfModel::best_available(config).unwrap();
    model.mixed_composition = Some(fixture.runtime(table.path()));
    let mut rows = group(Kind::ContainsLocallyMixed);
    rows[0].wall_time = 0.040;
    for _ in 0..20 {
        model.tune_with_fpms(&[rows.clone()]).unwrap();
    }
    // Raw fixture interpolation 14ms, calibrated cap 2 => 28ms. Training
    // labels have no online prediction and cannot count as validation.
    let route = model.estimate_forward_pass_with_route(&rows).unwrap();
    assert_eq!(route["latency_ms"], 28.0);
    assert_eq!(
        route["temporary_mixed_composition"]["paired_comparisons"],
        0
    );
    let (c, _) = checkpoint_with_stores(
        ForwardPassWorkerType::Aggregated,
        &[Kind::ContainsLocallyMixed],
        20,
        40.0,
    );
    let before = model.mixed_composition_state().unwrap();
    model.append_mixed_learned_checkpoint(c.path()).unwrap();
    assert_eq!(model.mixed_composition_state().unwrap(), before);
    assert!(model.append_mixed_learned_checkpoint(c.path()).is_err());
    assert_eq!(model.mixed_composition_state().unwrap(), before);
    // Merely becoming ready does not release the fallback.
    assert_eq!(
        model.estimate_forward_pass_time_ms(&rows).unwrap(),
        Some(28.0)
    );
    rows[0].wall_time = 0.028;
    model.tune_with_fpms(&[rows.clone()]).unwrap();
    assert_eq!(
        model.estimate_forward_pass_time_ms(&rows).unwrap(),
        Some(28.0)
    );
    rows[0].wall_time = 0.040;
    model.tune_with_fpms(&[rows.clone()]).unwrap();
    assert_eq!(
        model.estimate_forward_pass_time_ms(&rows).unwrap(),
        Some(40.0)
    );
    rows[0].wall_time = 0.028;
    model.tune_with_fpms(&[rows.clone()]).unwrap();
    let expected = 28.0;
    assert_eq!(
        model.estimate_forward_pass_time_ms(&rows).unwrap(),
        Some(expected)
    );
    assert_eq!(
        model
            .estimate_forward_pass_detailed(&rows)
            .unwrap()
            .latency_ms,
        Some(expected)
    );
    let prediction = model.estimate_forward_pass_with_route(&rows).unwrap();
    assert_eq!(prediction["latency_ms"], expected);
    assert_eq!(
        model.mixed_composition_route(&rows).unwrap().unwrap(),
        prediction["temporary_mixed_composition"]
    );
    assert_eq!(
        prediction["temporary_mixed_composition"]["paired_comparisons"],
        3
    );
    assert_eq!(
        model.ordinary_accepted_observations(Kind::ContainsLocallyMixed),
        3
    );
    assert_eq!(
        model
            .estimate_forward_pass_time_ms(&group(Kind::PureDecode))
            .unwrap(),
        Some(10.0)
    );
    let state = model.mixed_composition_state().unwrap().unwrap();
    let mut restored = model.clone();
    restored
        .restore_mixed_composition_state(state.clone())
        .unwrap();
    assert_eq!(
        restored.estimate_forward_pass_with_route(&rows).unwrap(),
        prediction
    );
    let mut corrupt = state.clone();
    corrupt["quality_gate"]["states"][5]["comparisons"] = 9999.into();
    assert!(restored.restore_mixed_composition_state(corrupt).is_err());
    assert_eq!(restored.mixed_composition_state().unwrap().unwrap(), state);
    rows[0].wall_time = f64::NAN;
    assert!(restored.tune_with_fpms(&[rows]).is_err());
    assert_eq!(restored.mixed_composition_state().unwrap().unwrap(), state);
}
