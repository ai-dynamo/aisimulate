# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from aisimulate_core.sdk import fpm_bundle, fpm_learned
from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

from . import test_fpm_mixed_composition_integration as mixed_fixtures
from .fpm_training_support import import_sklearn
from .fpm_training_support import isolate_training_dependencies as isolate_training_dependencies

phase_roots = mixed_fixtures.phase_roots
mixed_seed = mixed_fixtures.mixed_seed

pytestmark = pytest.mark.unit
import_sklearn()
pytest.importorskip("aisimulate_core._native")


def _iterations():
    return [
        [
            {
                "version": 1,
                "worker_id": "synthetic-test-worker",
                "dp_rank": 0,
                "counter_id": index,
                "wall_time": (1.0 + (index % 16) * 0.2) / 1000,
                "scheduled_requests": {
                    "num_decode_requests": index % 16 + 1,
                    "sum_decode_kv_tokens": (index % 16 + 1) * 100,
                },
            }
        ]
        for index in range(64)
    ]


@pytest.fixture
def identity():
    return {
        "model": "test-model",
        "system": "test-system",
        "backend": "vllm",
        "backend_version": "1",
        "worker_type": "decode",
    }


@pytest.fixture
def provenance():
    return {
        "anchor_kind": "b",
        "sources": [{"reference": "fixture:synthetic-training", "sha256": hashlib.sha256(b"training").hexdigest()}],
        "evaluation_sources": [],
    }


def _kwargs():
    return {"max_iter": 4, "max_leaf_nodes": 3, "min_samples_leaf": 2, "min_store_rows": 5}


@pytest.fixture
def checkpoint(tmp_path, identity):
    path = tmp_path / "checkpoint"
    fpm_learned.train(_iterations(), "decode", identity=identity, online_checkpoint_dir=path, **_kwargs())
    return path


def test_train_bundle_installs_portably_and_preserves_raw_seed_predictions(tmp_path, identity, provenance):
    output = tmp_path / "bundle"
    config = fpm_learned.train_bundle(
        _iterations(), "decode", output_dir=output, identity=identity, provenance=provenance, **_kwargs()
    )
    manifest = json.loads((output / "bundle.json").read_bytes())
    assert manifest["provenance"]["accepted_rows"] == {"pure_decode": 64}
    assert manifest["provenance"]["anchor_kind"] == "b"
    assert manifest["prior"] is None
    assert "bootstrap_recipe" not in manifest
    assert manifest["training_recipe"]["name"] == "sklearn.HistGradientBoostingRegressor"
    assert manifest["training_recipe"]["hyperparameters"]["max_iter"] == 4
    artifact = json.loads((output / "seed/model.json").read_bytes())
    checkpoint = json.loads((output / "seed/manifest.json").read_bytes())
    assert manifest["training_recipe"]["hyperparameters"] == artifact["metadata"]["hyperparameters"]
    assert manifest["training_recipe"]["trainer_version"] == checkpoint["sklearn_version"]
    original = RustForwardPassPerfModel.best_available(config)
    frozen = RustForwardPassPerfModel.best_available(
        fpm_learned.inspect_bundle(output / "bundle.json", adaptation="frozen")
    )
    expected = [original.estimate_forward_pass_time_ms(row) for row in _iterations()[:4]]
    assert expected == [frozen.estimate_forward_pass_time_ms(row) for row in _iterations()[:4]]
    assert all(value is not None for value in expected)
    before = original.export_learned_artifact()
    for row in _iterations():
        original.estimate_forward_pass_time_ms(row)
    assert original.export_learned_artifact() == before
    systems = tmp_path / "systems"
    installed = fpm_learned.install_bundle(output / "bundle.json", systems)
    shutil.rmtree(output)
    loaded = RustForwardPassPerfModel.best_available(installed)
    assert [loaded.estimate_forward_pass_time_ms(row) for row in _iterations()[:4]] == expected
    assert loaded.dependency_info()["files"]
    discovered = RustForwardPassPerfModel.best_available(
        {**identity, "estimation_mode": "fpm_learned", "systems_paths": [str(systems)]}
    )
    assert discovered.estimate_forward_pass_time_ms(_iterations()[0]) == expected[0]
    source = discovered.diagnostics()["provenance"]["config"]["estimator_config"]["fpm_learned"]
    assert source["source_sha256"] == installed["estimator_config"]["fpm_learned"]["source_sha256"]


def test_catalogue_default_switch_does_not_retarget_existing_configs(tmp_path, checkpoint, provenance):
    systems = tmp_path / "systems"
    first = tmp_path / "first"
    fpm_learned.export_bundle(checkpoint, first, provenance=provenance)
    pinned = fpm_learned.install_bundle(first / "bundle.json", systems)
    second = tmp_path / "second"
    fpm_learned.export_bundle(checkpoint, second, provenance={**provenance, "anchor_kind": "a"})
    second_config = fpm_learned.install_bundle(second / "bundle.json", systems)
    index = systems / "fpm_learned/index.json"
    assert [entry["default"] for entry in json.loads(index.read_bytes())["entries"]] == [True, False]
    fpm_learned.install_bundle(second / "bundle.json", systems, make_default=True)
    entries = json.loads(index.read_bytes())["entries"]
    assert len(entries) == 2
    assert [entry["default"] for entry in entries] == [False, True]
    assert (
        pinned["estimator_config"]["fpm_learned"]["bundle_path"]
        != second_config["estimator_config"]["fpm_learned"]["bundle_path"]
    )
    assert RustForwardPassPerfModel.best_available(pinned).estimate_forward_pass_time_ms(_iterations()[0]) is not None


@pytest.mark.parametrize("corruption", ["hash", "traversal", "symlink"])
def test_export_rejects_corrupt_or_escaping_checkpoint_without_publishing(tmp_path, checkpoint, provenance, corruption):
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    rows = checkpoint / manifest["stores"]["pure_decode"]["rows_file"]
    if corruption == "hash":
        rows.write_bytes(b"changed")
    elif corruption == "traversal":
        rows.rename(tmp_path / "outside.parquet")
        manifest["stores"]["pure_decode"]["rows_file"] = "../outside.parquet"
        manifest_path.write_text(json.dumps(manifest))
    else:
        outside = tmp_path / "outside.parquet"
        rows.rename(outside)
        rows.symlink_to(outside)
    with pytest.raises(ValueError):
        fpm_learned.export_bundle(checkpoint, tmp_path / "output", provenance=provenance)
    assert not (tmp_path / "output").exists()


def test_producer_refuses_training_evaluation_leakage_and_false_counts(tmp_path, checkpoint, provenance):
    for claims in (
        {**provenance, "evaluation_sources": provenance["sources"]},
        {**provenance, "accepted_rows": {"pure_decode": 9000}},
        {**provenance, "c_admission_lineage": {"last_admitted": 4}},
    ):
        with pytest.raises(ValueError):
            fpm_learned.export_bundle(checkpoint, tmp_path / "output", provenance=claims)
        assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("field", ["name", "trainer_version", "hyperparameters"])
def test_bundle_recipe_must_describe_the_offline_seed(tmp_path, checkpoint, provenance, field):
    output = tmp_path / "bundle"
    fpm_learned.export_bundle(checkpoint, output, provenance=provenance)
    manifest_path = output / "bundle.json"
    manifest = json.loads(manifest_path.read_bytes())
    if field == "hyperparameters":
        manifest["training_recipe"][field]["max_iter"] += 1
    else:
        manifest["training_recipe"][field] = "unrelated-trainer"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="trainer version|training recipe"):
        fpm_learned.inspect_bundle(manifest_path)


def test_install_does_not_overwrite_a_locked_or_corrupt_catalogue(tmp_path, checkpoint, provenance):
    fpm_learned.export_bundle(checkpoint, tmp_path / "bundle", provenance=provenance)
    source = tmp_path / "bundle/bundle.json"
    systems = tmp_path / "systems"
    fpm_learned.install_bundle(source, systems)
    index = systems / "fpm_learned/index.json"
    before = index.read_bytes()
    lock = index.parent / ".install.lock"
    lock.touch()
    with pytest.raises(FileExistsError, match="another bundle installation"):
        fpm_learned.install_bundle(source, systems)
    assert index.read_bytes() == before
    lock.unlink()
    catalogue = json.loads(before)
    catalogue["entries"][0]["default"] = False
    index.write_text(json.dumps(catalogue))
    with pytest.raises(ValueError, match="exactly one designated default"):
        fpm_learned.install_bundle(source, systems)
    assert not lock.exists()


def test_original_a_manifest_bytes_and_legacy_absolute_bindings_are_preserved(tmp_path, checkpoint):
    evidence = tmp_path / "original.jsonl"
    evidence.write_text("source-bytes-for-test")
    original = tmp_path / "original-a.json"
    payload = {
        "provenance": {"sources": [{"path": str(evidence), "sha256": fpm_learned._file_sha256(evidence)}]},
        "seed_artifacts": [
            {
                "path": str(checkpoint / "model.json"),
                "sha256": fpm_learned._file_sha256(checkpoint / "model.json"),
                "checkpoint_manifest_sha256": fpm_learned._file_sha256(checkpoint / "manifest.json"),
            }
        ],
    }
    original.write_text(json.dumps(payload, indent=3))
    staging = tmp_path / "staging"
    staging.mkdir()
    dependencies = []
    prior = fpm_bundle._copy_prior(
        {"original_a": {"manifest_path": str(original), "source_sha256": fpm_learned._file_sha256(original)}},
        staging,
        dependencies,
    )
    assert (staging / prior["original_a"]["manifest_path"]).read_bytes() == original.read_bytes()
    assert {item["original_path"] for item in prior["dependency_remap"]} == {
        str(evidence),
        str(checkpoint / "model.json"),
        str(checkpoint / "manifest.json"),
    }
    shutil.rmtree(checkpoint)
    evidence.unlink()
    for entry in dependencies:
        assert fpm_learned._file_sha256(staging / entry["path"]) == entry["sha256"]


def test_existing_checkpoint_cli_emits_constructible_pinned_configuration(tmp_path, checkpoint, provenance, capsys):
    claims = tmp_path / "provenance.json"
    claims.write_text(json.dumps(provenance))
    output = tmp_path / "bundle"
    assert (
        fpm_learned.main(
            ["bundle", "--checkpoint", str(checkpoint), "--output-dir", str(output), "--provenance", str(claims)]
        )
        == 0
    )
    resolved = json.loads(capsys.readouterr().out)
    assert RustForwardPassPerfModel.best_available(resolved).estimate_forward_pass_time_ms(_iterations()[0]) is not None
    assert (
        fpm_learned.main(
            ["install-bundle", "--bundle", str(output / "bundle.json"), "--systems-root", str(tmp_path / "systems")]
        )
        == 0
    )
    installed = json.loads(capsys.readouterr().out)
    assert Path(installed["estimator_config"]["fpm_learned"]["bundle_path"]).is_file()


def test_inspection_checks_expected_identity_and_retains_caller_controls(tmp_path, checkpoint, identity, provenance):
    output = tmp_path / "bundle"
    fpm_learned.export_bundle(checkpoint, output, provenance=provenance)
    request = {
        **identity,
        "estimation_mode": "fpm_learned",
        "estimator_config": {"fpm_learned": {"recent_window": 17}},
    }
    before = copy.deepcopy(request)
    resolved = fpm_learned.inspect_bundle(output / "bundle.json", config=request)
    assert request == before
    assert resolved["estimator_config"]["fpm_learned"]["recent_window"] == 17
    frozen_request = {**request, "estimator_config": {"fpm_learned": {"adaptation": "frozen"}}}
    frozen = fpm_learned.inspect_bundle(output / "bundle.json", config=frozen_request)
    assert frozen["estimator_config"]["fpm_learned"]["adaptation"] == "frozen"
    with pytest.raises(ValueError, match="identity"):
        fpm_learned.inspect_bundle(output / "bundle.json", config={**request, "tp": 2})


def test_legacy_normalized_artifact_can_be_replaced_by_clearing_its_hash(tmp_path, checkpoint, identity):
    request = {
        **identity,
        "estimation_mode": "fpm_learned",
        "estimator_config": {"fpm_learned": {"artifact_path": str(checkpoint / "model.json"), "adaptation": "frozen"}},
    }
    original = RustForwardPassPerfModel.best_available(request)
    try:
        resolved = original.diagnostics()["provenance"]["config"]
        learned = resolved["estimator_config"]["fpm_learned"]
        assert "resolved_source" not in learned
        assert "resolved_fallback" not in learned
        assert original.dependency_info()["files"]
        expected = original.estimate_forward_pass_time_ms(_iterations()[0])
        artifact = original.export_learned_artifact()
    finally:
        original.close()
    artifact["stores"]["pure_decode"]["baseline"] += 1.0
    replacement_path = tmp_path / "replacement.json"
    replacement_path.write_text(json.dumps(artifact))
    learned["artifact_path"] = str(replacement_path)
    with pytest.raises(ValueError, match="checksum mismatch"):
        RustForwardPassPerfModel.best_available(resolved)
    learned.pop("source_sha256")
    replacement = RustForwardPassPerfModel.best_available(resolved)
    try:
        assert replacement.estimate_forward_pass_time_ms(_iterations()[0]) != expected
    finally:
        replacement.close()


def test_legacy_mixed_policy_can_be_added_after_ordinary_normalization(mixed_seed):
    ordinary = mixed_fixtures._config(mixed_seed, enabled=False)
    resolved = RustForwardPassPerfModel.normalize_config(ordinary)
    learned = resolved["estimator_config"]["fpm_learned"]
    assert "resolved_source" not in learned
    assert "resolved_fallback" not in learned
    mixed = mixed_fixtures._config(mixed_seed)
    resolved["estimator_config"]["temporary_mixed_composition"] = copy.deepcopy(
        mixed["estimator_config"]["temporary_mixed_composition"]
    )
    normalized = RustForwardPassPerfModel.normalize_config(resolved)
    model = RustForwardPassPerfModel.best_available(normalized)
    try:
        assert model.estimate_forward_pass_time_ms(mixed_fixtures._iteration()) == 4.0
        assert model.dependency_info()["fallback"]
        assert len(model.dependency_info()["files"]) >= 3
        assert "resolved_source" not in model.diagnostics()["provenance"]["config"]["estimator_config"]["fpm_learned"]
    finally:
        model.close()


def test_bundled_prior_preserves_native_dependencies_and_original_evidence(tmp_path, mixed_seed):
    config, seed, original_a = mixed_seed
    legacy = mixed_fixtures._config(mixed_seed)
    reference = RustForwardPassPerfModel.best_available(legacy)
    expected = reference.estimate_forward_pass_time_ms(mixed_fixtures._iteration())
    request = mixed_fixtures._config(mixed_seed, enabled=False)
    prior = {
        "original_a": original_a,
        "calibration": legacy["estimator_config"]["temporary_mixed_composition"]["calibration"],
    }
    output = tmp_path / "bundle"
    resolved = fpm_learned.export_bundle(
        seed,
        output,
        config=request,
        prior=prior,
        provenance={
            "anchor_kind": "a",
            "sources": [{"reference": "fixture:original-a", "sha256": original_a["source_sha256"]}],
            "evaluation_sources": [],
        },
    )
    fallback = resolved["estimator_config"]["fpm_learned"]["resolved_fallback"]
    assert fallback["kind"] == "mixed_interpolation_v1"
    assert fallback["native"]
    assert fallback["dependency_remap"]
    # Original source/checkpoint bytes are now owned by the bundle. Native
    # interpolation data remains an explicitly recorded external dependency.
    Path(original_a["manifest_path"]).unlink()
    shutil.rmtree(seed)
    loaded = RustForwardPassPerfModel.best_available(resolved)
    assert loaded.estimate_forward_pass_time_ms(mixed_fixtures._iteration()) == expected == 4.0
    dependencies = loaded.dependency_info()
    assert dependencies["systems_paths"] == config["systems_paths"]
    assert dependencies["fallback"]
    assert all(Path(item["path"]).exists() for item in dependencies["files"])
    reloaded = RustForwardPassPerfModel.best_available(json.loads(json.dumps(resolved)))
    assert reloaded.estimate_forward_pass_time_ms(mixed_fixtures._iteration()) == expected


def test_train_cli_validates_real_source_hashes_and_writes_bundle_config(tmp_path, identity, capsys):
    source = tmp_path / "train.jsonl"
    source.write_text("".join(json.dumps(group[0]) + "\n" for group in _iterations()))
    claims = tmp_path / "claims.json"
    claims.write_text(
        json.dumps(
            {
                "anchor_kind": "a",
                "sources": [{"reference": str(source), "sha256": fpm_learned._file_sha256(source)}],
                "evaluation_sources": [],
            }
        )
    )
    identity_file = tmp_path / "identity.json"
    identity_file.write_text(json.dumps(identity))
    output = tmp_path / "bundle"
    saved_config = tmp_path / "prediction.json"
    args = [
        "train",
        "--fpm",
        str(source),
        "--worker-type",
        "decode",
        "--identity",
        str(identity_file),
        "--out",
        str(tmp_path / "artifact.json"),
        "--holdout-frac",
        "0",
        "--bundle-output-dir",
        str(output),
        "--provenance",
        str(claims),
        "--prediction-config-out",
        str(saved_config),
        "--max-iter",
        "3",
        "--max-leaf-nodes",
        "3",
        "--min-samples-leaf",
        "2",
        "--min-store-rows",
        "5",
    ]
    assert fpm_learned.main(args) == 0
    config = json.loads(saved_config.read_bytes())
    assert RustForwardPassPerfModel.best_available(config).estimate_forward_pass_time_ms(_iterations()[0]) is not None
    assert "wrote validated bundle" in capsys.readouterr().out
    shutil.rmtree(output)
    saved_config.unlink()
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="digests must exactly match"):
        fpm_learned.main(args)
    assert not output.exists()
