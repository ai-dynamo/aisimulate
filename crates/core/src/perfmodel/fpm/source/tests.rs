// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use serde_json::json;

fn config() -> ForwardPassPerfModelConfig {
    serde_json::from_value(json!({
        "model":"test-model", "system":"test-system", "backend":"vllm",
        "backend_version":"1", "worker_type":"decode", "estimation_mode":"fpm_learned",
        "estimator_config":{"fpm_learned":{"adaptation":"frozen"}}
    }))
    .unwrap()
}
fn write_json(path: &Path, value: &serde_json::Value) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
}
fn bundle(root: &Path, config: &ForwardPassPerfModelConfig) -> PathBuf {
    let identity = super::super::learned_online::canonical_learned_identity(config).unwrap();
    let artifact_path = root.join("seed.json");
    write_json(
        &artifact_path,
        &json!({
            "schema":"aic_fpm_learned_forward_perf", "schema_version":1,
            "worker_type":"decode", "features":["num_decode_requests"], "target":"log_ms",
            "stores":{"pure_decode":{"baseline":0.0,"trees":[]}},
            "metadata":{"identity":identity,"train_rows":{"pure_decode":20},
                "model":"fixture-trainer","hyperparameters":{"max_iter":1}}
        }),
    );
    let artifact = dependency(&artifact_path, "learned_artifact").unwrap();
    let path = root.join("bundle.json");
    write_json(
        &path,
        &json!({
            "kind":"aisimulate_learned_bundle", "schema_version":1,
            "producer_version":"test-v1", "recipe_version":"test-v1", "identity":identity,
            "seed":{"kind":"artifact", "path":"seed.json", "sha256":artifact.sha256,
                "adaptations":["frozen"],"covered_stores":["pure_decode"]},
            "training_recipe":{"name":"fixture-trainer","features":["num_decode_requests"],
                "target":"log_ms","trainer_version":"test-v1","hyperparameters":{"max_iter":1}},
            "provenance":{"anchor_kind":"b","accepted_rows":{"pure_decode":20},
                "sources":[{"reference":"fixture:accepted","sha256":"a".repeat(64)}],
                "evaluation_sources":[],"c_admission_lineage":null},
            "dependencies":[{"path":"seed.json","byte_length":artifact.byte_length,
                "sha256":artifact.sha256,"purpose":"learned_artifact"}]
        }),
    );
    path
}
fn mutate(path: &Path, edit: impl FnOnce(&mut serde_json::Value)) {
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    edit(&mut value);
    write_json(path, &value);
}

#[test]
fn explicit_bundle_pins_payload_and_round_trips() {
    let directory = tempfile::tempdir().unwrap();
    let mut request = config();
    let path = bundle(directory.path(), &request);
    request.estimator_config.fpm_learned.source.bundle_path = Some(path.clone());
    let resolved = request.resolve().unwrap();
    let learned = &resolved.estimator_config.fpm_learned;
    let pin = learned.source.resolved_source.as_ref().unwrap();
    assert_eq!(pin.kind, LearnedSourceKind::Bundle);
    assert_eq!(pin.dependencies.len(), 2);
    assert_eq!(learned.source_sha256, pin.sha256);
    let runtime = learned.runtime_source_config();
    assert_eq!(
        runtime.artifact_path,
        Some(directory.path().join("seed.json"))
    );
    assert_eq!(runtime.source_sha256, pin.seed_sha256);
    assert_eq!(runtime.source, LearnedSourceConfig::default());
    let encoded = serde_json::to_value(&resolved).unwrap();
    assert!(
        encoded
            .pointer("/estimator_config/fpm_learned/bundle_path")
            .is_some()
    );
    assert!(
        encoded
            .pointer("/estimator_config/fpm_learned/source")
            .is_none()
    );
    let restored: ForwardPassPerfModelConfig = serde_json::from_value(encoded).unwrap();
    assert_eq!(restored.resolve().unwrap(), resolved);
    // Source pins bind bytes, not the continued existence of the path alone.
    mutate(&path, |bundle| {
        bundle["producer_version"] = json!("changed")
    });
    assert!(
        resolved
            .resolve()
            .unwrap_err()
            .to_string()
            .contains("sha256")
    );
}

#[test]
fn bundle_rejects_corruption_identity_abi_and_capability_conflicts() {
    for case in [
        "digest",
        "length",
        "identity",
        "features",
        "recipe",
        "stores",
        "capability",
        "leakage",
    ] {
        let directory = tempfile::tempdir().unwrap();
        let mut request = config();
        let path = bundle(directory.path(), &request);
        mutate(&path, |value| match case {
            "digest" => value["dependencies"][0]["sha256"] = json!("0".repeat(64)),
            "length" => value["dependencies"][0]["byte_length"] = json!(1),
            "identity" => value["identity"]["tp"] = json!(2),
            "features" => value["training_recipe"]["features"] = json!(["sum_decode_kv_tokens"]),
            "recipe" => value["training_recipe"]["hyperparameters"]["max_iter"] = json!(150),
            "stores" => value["seed"]["covered_stores"] = json!(["pure_decode", "pure_decode"]),
            "capability" => value["seed"]["adaptations"] = json!(["online"]),
            "leakage" => {
                value["provenance"]["evaluation_sources"] = value["provenance"]["sources"].clone()
            }
            _ => unreachable!(),
        });
        request.estimator_config.fpm_learned.source.bundle_path = Some(path);
        assert!(request.resolve().is_err(), "must reject {case}");
    }
}

#[test]
fn source_absence_is_pinned_even_if_catalogue_appears() {
    let directory = tempfile::tempdir().unwrap();
    let mut request = config();
    request.estimator_config.fpm_learned.adaptation = LearnedAdaptation::Online;
    request.systems_paths = vec![directory.path().to_owned()];
    assert!(matches!(
        resolve(&mut request),
        Err(AicError::LearnedSourceUnavailable { .. })
    ));
    assert_eq!(
        request.estimator_config.fpm_learned.source.resolved_source,
        Some(ResolvedLearnedSource::absent())
    );
    write_json(
        &directory.path().join("fpm_learned/index.json"),
        &json!({"invalid":"catalogue"}),
    );
    // A resolved absence bypasses discovery, including a newly corrupt index.
    assert!(matches!(
        resolve(&mut request),
        Err(AicError::LearnedSourceUnavailable { .. })
    ));
    request.estimator_config.fpm_learned.source.resolved_source = None;
    assert!(resolve(&mut request).is_err());
}

#[test]
fn frozen_and_explicit_sources_do_not_discover_catalogues() {
    let directory = tempfile::tempdir().unwrap();
    write_json(
        &directory.path().join("fpm_learned/index.json"),
        &json!({"invalid":"catalogue"}),
    );
    let mut request = config();
    request.systems_paths = vec![directory.path().to_owned()];
    // Explicit frozen source remains supported, irrespective of an index.
    let artifact = directory.path().join("raw.json");
    std::fs::write(&artifact, b"raw source binding").unwrap();
    request.estimator_config.fpm_learned.artifact_path = Some(artifact);
    request.resolve().unwrap();
    let mut request = config();
    request.systems_paths = vec![directory.path().to_owned()];
    assert!(matches!(
        resolve(&mut request),
        Err(AicError::InvalidEngineConfig(_))
    ));
    assert_eq!(
        request.estimator_config.fpm_learned.source.resolved_source,
        Some(ResolvedLearnedSource::absent())
    );
}

#[test]
fn catalogue_requires_one_default_and_honors_ordered_roots() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    let mut request = config();
    request.estimator_config.fpm_learned.adaptation = LearnedAdaptation::Online;
    request.systems_paths = vec![first.path().to_owned(), second.path().to_owned()];
    let identity = super::super::learned_online::canonical_learned_identity(&request).unwrap();
    let entry = json!({"identity":identity,"recipe_version":"recipe","bundle_path":"bundle.json","sha256":"a".repeat(64),"default":true});
    let index = first.path().join("fpm_learned/index.json");
    write_json(
        &index,
        &json!({"kind":"aisimulate_learned_catalogue","schema_version":1,"entries":[entry.clone(),entry.clone()]}),
    );
    assert!(
        catalogue_source(&request)
            .unwrap_err()
            .to_string()
            .contains("exactly one")
    );
    mutate(&index, |value| {
        value["entries"] = json!([entry]);
        value["entries"][0]["default"] = json!(false);
    });
    assert!(
        catalogue_source(&request)
            .unwrap_err()
            .to_string()
            .contains("exactly one")
    );
    // A matching, broken first-root entry is not skipped for a later root.
    mutate(&index, |value| value["entries"][0]["default"] = json!(true));
    assert!(catalogue_source(&request).is_err());
}

#[test]
fn stable_and_legacy_fallback_inputs_are_ambiguous_even_for_auto() {
    let mut value = serde_json::to_value(config()).unwrap();
    value["estimator_config"]["fpm_learned"]["fallback"] = json!("auto");
    value["estimator_config"]["temporary_mixed_composition"] = json!({
        "scenario":"a","mode":"compose","original_a":{"manifest_path":"a.json"}
    });
    let error = serde_json::from_value::<ForwardPassPerfModelConfig>(value).unwrap_err();
    assert!(error.to_string().contains("cannot both"));
}

#[test]
fn relative_dependencies_cannot_escape_bundle_root() {
    let root = tempfile::tempdir().unwrap();
    assert!(relative(root.path(), Path::new("../outside")).is_err());
    assert!(relative(root.path(), Path::new("/absolute")).is_err());
    #[cfg(unix)]
    {
        let outside = tempfile::NamedTempFile::new().unwrap();
        std::os::unix::fs::symlink(outside.path(), root.path().join("escape")).unwrap();
        assert!(
            relative(root.path(), Path::new("escape"))
                .unwrap_err()
                .to_string()
                .contains("escapes")
        );
    }
}

#[test]
fn catalogue_default_selection_is_pinned_before_the_designation_changes() {
    let directory = tempfile::tempdir().unwrap();
    let mut request = config();
    request.estimator_config.fpm_learned.adaptation = LearnedAdaptation::Online;
    request.systems_paths = vec![directory.path().to_owned()];
    let root = directory.path().join("fpm_learned/seed");
    let path = bundle(&root, &request);
    let identity = super::super::learned_online::canonical_learned_identity(&request).unwrap();
    let artifact = dependency(&root.join("seed.json"), "learned_artifact").unwrap();
    // Source resolution treats row payloads as opaque authenticated bytes. The
    // online checkpoint loader separately validates their parquet schema.
    std::fs::write(root.join("rows.parquet"), b"source-resolution-test-rows").unwrap();
    let rows = dependency(&root.join("rows.parquet"), "training_rows").unwrap();
    write_json(
        &root.join("manifest.json"),
        &json!({
            "schema":"aisimulate_fpm_training_checkpoint", "schema_version":1,"sklearn_version":"test-v1",
            "identity":identity,"artifact_file":"seed.json","artifact_sha256":artifact.sha256,
            "stores":{"pure_decode":{"row_count":20,"rows_file":"rows.parquet","rows_sha256":rows.sha256}}
        }),
    );
    let checkpoint = dependency(&root.join("manifest.json"), "checkpoint_manifest").unwrap();
    mutate(&path, |value| {
        value["seed"]["kind"] = json!("training_checkpoint");
        // Put the bundle manifest one directory above its checkpoint. Every
        // relative path remains normal and the checkpoint directory is nonempty.
        value["seed"]["path"] = json!("seed");
        value["seed"]["sha256"] = json!(checkpoint.sha256);
        value["seed"]["adaptations"] = json!(["online", "frozen", "causal_selector"]);
        value["dependencies"] = json!([
            {"path":"seed/seed.json","byte_length":artifact.byte_length,"sha256":artifact.sha256,"purpose":"learned_artifact"},
            {"path":"seed/rows.parquet","byte_length":rows.byte_length,"sha256":rows.sha256,"purpose":"training_rows"},
            {"path":"seed/manifest.json","byte_length":checkpoint.byte_length,"sha256":checkpoint.sha256,"purpose":"checkpoint_manifest"}
        ]);
    });
    let bundle_path = directory.path().join("fpm_learned/bundle.json");
    std::fs::rename(&path, &bundle_path).unwrap();
    let bundle_hash = dependency(&bundle_path, "bundle_manifest").unwrap().sha256;
    let index = directory.path().join("fpm_learned/index.json");
    write_json(
        &index,
        &json!({"kind":"aisimulate_learned_catalogue","schema_version":1,
        "entries":[{"identity":identity,"recipe_version":"test-v1","bundle_path":"bundle.json","sha256":bundle_hash,"default":true}]}),
    );
    resolve(&mut request).unwrap();
    assert_eq!(
        request.estimator_config.fpm_learned.source.bundle_path,
        Some(bundle_path)
    );
    assert_eq!(
        request
            .estimator_config
            .fpm_learned
            .runtime_source_config()
            .checkpoint_path,
        Some(root)
    );
    let pin = request
        .estimator_config
        .fpm_learned
        .source
        .resolved_source
        .clone();
    mutate(&index, |value| {
        value["entries"][0]["default"] = json!(false)
    });
    resolve(&mut request).unwrap();
    assert_eq!(
        request.estimator_config.fpm_learned.source.resolved_source,
        pin
    );
}

#[test]
fn training_identity_does_not_require_or_discover_its_future_source() {
    let mut request = config();
    request.estimator_config.fpm_learned.source.bundle_path =
        Some("/missing/future-bundle.json".into());
    request.systems_paths = vec!["/missing/catalogue".into()];
    let expected = super::super::learned_online::canonical_learned_identity(&request).unwrap();
    assert_eq!(request.resolve_learned_identity().unwrap(), expected);
    // Source-free online identity, the ordinary input to the bundle producer.
    let mut request = config();
    request.estimator_config.fpm_learned.adaptation = LearnedAdaptation::Online;
    assert_eq!(request.resolve_learned_identity().unwrap(), expected);
}

#[test]
fn legacy_explicit_source_replacement_keeps_the_historical_hash_contract() {
    let directory = tempfile::tempdir().unwrap();
    let first = directory.path().join("first.json");
    let second = directory.path().join("second.json");
    std::fs::write(&first, b"first seed").unwrap();
    std::fs::write(&second, b"replacement seed").unwrap();
    let mut request = config();
    request.estimator_config.fpm_learned.artifact_path = Some(first);
    let mut resolved = request.resolve().unwrap();
    assert_eq!(
        resolved.estimator_config.fpm_learned.source,
        LearnedSourceConfig::default()
    );
    let first_hash = resolved.estimator_config.fpm_learned.source_sha256.clone();
    resolved.estimator_config.fpm_learned.artifact_path = Some(second);
    assert!(
        resolved
            .clone()
            .resolve()
            .unwrap_err()
            .to_string()
            .contains("checksum mismatch")
    );
    resolved.estimator_config.fpm_learned.source_sha256 = None;
    let replacement = resolved.resolve().unwrap();
    assert_ne!(
        replacement.estimator_config.fpm_learned.source_sha256,
        first_hash
    );
    assert_eq!(
        replacement.estimator_config.fpm_learned.source,
        LearnedSourceConfig::default()
    );
}

#[test]
fn a_normalized_legacy_source_can_add_a_legacy_mixed_policy() {
    let directory = tempfile::tempdir().unwrap();
    let artifact = directory.path().join("seed.json");
    let original_a = directory.path().join("original-a.json");
    std::fs::write(&artifact, b"seed").unwrap();
    write_json(
        &original_a,
        &json!({"seed_artifacts":[],"provenance":{"sources":[]}}),
    );
    let mut request = config();
    request.worker_type = super::super::ForwardPassWorkerType::Aggregated;
    request.estimator_config.fpm_learned.artifact_path = Some(artifact);
    let mut normalized = serde_json::to_value(request.resolve().unwrap()).unwrap();
    normalized["estimator_config"]["temporary_mixed_composition"] = json!({
        "scenario":"a","mode":"compose","original_a":{"manifest_path":original_a}
    });
    let resolved = serde_json::from_value::<ForwardPassPerfModelConfig>(normalized)
        .unwrap()
        .resolve()
        .unwrap();
    assert!(
        resolved
            .estimator_config
            .temporary_mixed_composition
            .is_some()
    );
    assert_eq!(
        resolved.estimator_config.fpm_learned.source,
        LearnedSourceConfig::default()
    );
    let mut captured = resolved.clone();
    pin_legacy_source(&mut captured).unwrap();
    assert!(
        captured
            .estimator_config
            .temporary_mixed_composition
            .is_none()
    );
    let source = captured
        .estimator_config
        .fpm_learned
        .source
        .resolved_source
        .as_ref()
        .unwrap();
    assert_eq!(source.kind, LearnedSourceKind::Artifact);
    assert_eq!(source.dependencies.len(), 2);
    assert!(
        captured
            .estimator_config
            .fpm_learned
            .source
            .resolved_fallback
            .as_ref()
            .unwrap()
            .policy
            .is_some()
    );
    assert!(
        resolved
            .estimator_config
            .temporary_mixed_composition
            .is_some()
    );
}

#[test]
fn supplied_source_and_absence_pins_cannot_be_discarded_by_legacy_edits() {
    let directory = tempfile::tempdir().unwrap();
    let first = directory.path().join("first.json");
    let second = directory.path().join("second.json");
    std::fs::write(&first, b"first seed").unwrap();
    std::fs::write(&second, b"replacement seed").unwrap();
    let mut pinned = config();
    pinned.estimator_config.fpm_learned.artifact_path = Some(first);
    pin_legacy_source(&mut pinned).unwrap();
    let mut changed = pinned.clone();
    changed.estimator_config.fpm_learned.artifact_path = Some(second);
    changed.estimator_config.fpm_learned.source_sha256 = None;
    assert!(
        changed
            .resolve()
            .unwrap_err()
            .to_string()
            .contains("conflicts with resolved source")
    );
    let mut mixed = serde_json::to_value(pinned).unwrap();
    mixed["estimator_config"]["temporary_mixed_composition"] = json!({
        "scenario":"a","mode":"compose","original_a":{"manifest_path":"new-policy.json"}
    });
    assert!(
        serde_json::from_value::<ForwardPassPerfModelConfig>(mixed)
            .unwrap_err()
            .to_string()
            .contains("cannot both")
    );

    let mut absent = config();
    absent.estimator_config.fpm_learned.artifact_path = Some(directory.path().join("first.json"));
    absent.estimator_config.fpm_learned.source.resolved_source =
        Some(ResolvedLearnedSource::absent());
    assert!(
        absent
            .resolve()
            .unwrap_err()
            .to_string()
            .contains("absence conflicts")
    );
}

#[test]
fn discovery_absence_is_distinct_from_missing_or_corrupt_explicit_sources() {
    let directory = tempfile::tempdir().unwrap();
    let mut request = config();
    request.estimator_config.fpm_learned.adaptation = LearnedAdaptation::Online;
    request.estimator_config.correction.enabled = false;
    request.systems_paths = vec![directory.path().to_owned()];
    assert!(matches!(
        request.clone().resolve(),
        Err(AicError::LearnedSourceUnavailable { .. })
    ));
    request.estimator_config.fpm_learned.checkpoint_path = Some(directory.path().join("missing"));
    assert!(matches!(
        request.clone().resolve(),
        Err(AicError::Io { .. })
    ));
    request.estimator_config.fpm_learned.checkpoint_path = None;
    request.estimator_config.fpm_learned.source.source_policy = LearnedSourcePolicy::ExplicitOnly;
    assert!(matches!(
        request.resolve(),
        Err(AicError::InvalidEngineConfig(_))
    ));
}
