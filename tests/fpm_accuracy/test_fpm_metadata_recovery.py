# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reconstructed schema-6 sidecars stay source-bound during native staging."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.fpm_accuracy.exceptions import ConfigurationError
from scripts.fpm_accuracy.models import aic_fpm_database

IDENTITY = {
    "model_path": "MiniMaxAI/MiniMax-M2.7",
    "system": "h200_sxm",
    "backend": "vllm",
    "backend_version": "0.25.1",
    "tp": 4,
    "pp": 1,
    "dp": 1,
    "moe_tp": 1,
    "moe_ep": 4,
    "cp": 1,
    "parallel_strategy": "tep",
    "weight_quantization": "fp8_block",
    "kv_cache_dtype": "fp8",
}
FACTS = {
    "coordinate_system": "iteration_totals_balanced_v1",
    "measurement_policy": "dynamo_native_single_sample_v1",
    "measurement_repeats": 1,
    "warmup_repeats": 0,
    "system": "h200_sxm",
    "backend": "vllm",
    "backend_version": "0.25.1",
}
ROW = {
    **IDENTITY,
    "partition_policy": "balanced_v1",
    "batch_size": 1,
    "total_prefill_tokens": 1,
    "total_kv_read_tokens": 0,
    "measurement_policy": "dynamo_native_single_sample_v1",
    "measurement_repeats": 1,
    "warmup_repeats": 0,
    "latency_ms": 8.375,
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def source(tmp_path, monkeypatch):
    parquet = tmp_path / "fpm.kv-on.parquet"
    sidecar = tmp_path / "fpm.kv-on.metadata.json"
    overlays = []

    def overlay(root, engine_config):
        directory = root / "data/h200_sxm/vllm/0.25.1"
        directory.mkdir(parents=True)
        overlays.append(root)
        return directory

    monkeypatch.setattr(aic_fpm_database, "_prepare_system_overlay", overlay)

    def write(*, rows=None, changes=None, omit=()):
        rows = [ROW, {**ROW, "total_prefill_tokens": 2, "latency_ms": 9.25}] if rows is None else rows
        pq.write_table(pa.Table.from_pylist(rows), parquet)
        metadata = {
            "schema_name": "aic_fpm_forward_perf",
            "schema_version": 6,
            "configuration_selector": IDENTITY,
            "metadata_provenance": "reconstructed_from_parquet",
            "notes": "Upstream metadata is unavailable; identity comes from recorded rows.",
            "producer_revision": None,
            "aisim_commit": None,
            "backend": "vllm",
            "row_count": len(rows),
            "parquet_sha256": digest(parquet),
            **(changes or {}),
        }
        for name in omit:
            metadata.pop(name)
        sidecar.write_text(json.dumps(metadata))
        return SimpleNamespace(
            artifact_id="reconstructed-h200-tep4",
            local_path=parquet,
            local_metadata_path=sidecar,
            sha256=digest(parquet),
            metadata_sha256=digest(sidecar),
        )

    return write, overlays


def test_recovery_stages_only_verified_declarations_and_preserves_originals(source):
    write, overlays = source
    artifact = write()
    original_parquet = artifact.local_path.read_bytes()
    original_sidecar = artifact.local_metadata_path.read_bytes()
    original = json.loads(original_sidecar)
    prepared = aic_fpm_database.prepare_aic_fpm_database({}, artifact)
    root = prepared.systems_root
    try:
        staged = next(root.rglob("fpm_forward_perf.metadata.json"))
        parquet = next(root.rglob("fpm_forward_perf.parquet"))
        assert not staged.is_symlink()
        assert parquet.is_symlink() and parquet.resolve() == artifact.local_path
        completed = json.loads(staged.read_text())
        assert completed == {**original, **FACTS}
        assert completed["producer_revision"] is None and completed["aisim_commit"] is None
        assert completed["metadata_provenance"] == "reconstructed_from_parquet"
        assert prepared.diagnostics["fpm_metadata_recovery"] == {
            "policy": "schema6_reconstructed_row_contract_v1",
            "source_metadata_sha256": artifact.metadata_sha256,
            "staged_metadata_sha256": digest(staged),
            "recovered_fields": {name: value for name, value in FACTS.items() if name not in original},
        }
        assert prepared.diagnostics["fpm_parquet_sha256"] == artifact.sha256
        assert artifact.local_path.read_bytes() == original_parquet
        assert artifact.local_metadata_path.read_bytes() == original_sidecar
    finally:
        prepared.close()
    assert overlays == [root] and not root.exists()
    assert artifact.local_path.is_file() and artifact.local_metadata_path.is_file()


@pytest.mark.parametrize("provenance", ["reconstructed_from_parquet", "collector"])
def test_complete_metadata_is_linked_without_rewriting_or_recovery_claim(source, provenance):
    write, _ = source
    artifact = write(changes={**FACTS, "metadata_provenance": provenance})
    before = artifact.local_metadata_path.read_bytes()
    prepared = aic_fpm_database.prepare_aic_fpm_database({}, artifact)
    try:
        staged = next(prepared.systems_root.rglob("fpm_forward_perf.metadata.json"))
        assert staged.is_symlink() and staged.resolve() == artifact.local_metadata_path
        assert staged.read_bytes() == before
        assert "fpm_metadata_recovery" not in prepared.diagnostics
    finally:
        prepared.close()


@pytest.mark.parametrize("target", ["parquet", "sidecar", "manifest_parquet", "manifest_sidecar"])
def test_source_mutation_or_incorrect_manifest_hash_is_rejected_before_staging(source, target):
    write, overlays = source
    artifact = write()
    if target == "parquet":
        artifact.local_path.write_bytes(artifact.local_path.read_bytes() + b"changed")
    elif target == "sidecar":
        artifact.local_metadata_path.write_text(artifact.local_metadata_path.read_text() + " ")
    elif target == "manifest_parquet":
        artifact.sha256 = "0" * 64
    else:
        artifact.metadata_sha256 = "0" * 64
    with pytest.raises(ConfigurationError, match="pinned hashes"):
        aic_fpm_database.prepare_aic_fpm_database({}, artifact)
    assert not overlays


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"parquet_sha256": "0" * 64}, "pinned hashes"),
        ({"schema_name": "other"}, "schema-6"),
        ({"row_count": 1}, "row_count"),
        ({"row_count": True}, "row_count"),
        ({"row_count": 0}, "row_count"),
        ({"coordinate_system": "other"}, "coordinate_system"),
        ({"measurement_policy": "median_of_3"}, "measurement_policy"),
        ({"measurement_repeats": True}, "measurement_repeats"),
        ({"system": "b200_sxm"}, "system"),
        ({"configuration_selector": {}}, "configuration selector"),
        ({"configuration_selector": {**IDENTITY, "tp": 8}}, "selector conflicts with row tp"),
        ({"configuration_selector": {**IDENTITY, "unrecorded_identity": "guess"}}, "contract columns"),
    ],
)
def test_conflicting_or_incomplete_metadata_never_gets_repaired(source, changes, message):
    write, overlays = source
    artifact = write(changes=changes)
    original = artifact.local_metadata_path.read_bytes()
    with pytest.raises(ConfigurationError, match=message):
        aic_fpm_database.prepare_aic_fpm_database({}, artifact)
    assert not overlays and artifact.local_metadata_path.read_bytes() == original


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"model_path": "Other/Model"}, "model_path"),
        ({"backend_version": "0.26.0"}, "backend_version"),
        ({"partition_policy": "unknown"}, "balanced_v1"),
        ({"batch_size": 0}, "coordinates"),
        ({"total_kv_read_tokens": -1}, "coordinates"),
        ({"measurement_policy": "median_of_3"}, "single-sample"),
        ({"measurement_repeats": 3}, "single-sample"),
        ({"warmup_repeats": 5}, "single-sample"),
    ],
)
def test_every_row_must_prove_the_recovered_contract(source, changes, message):
    write, overlays = source
    artifact = write(rows=[ROW, {**ROW, **changes}])
    with pytest.raises(ConfigurationError, match=message):
        aic_fpm_database.prepare_aic_fpm_database({}, artifact)
    assert not overlays


def test_validation_continues_beyond_the_first_bounded_batch(source):
    write, overlays = source
    artifact = write(rows=[ROW] * 4096 + [{**ROW, "tp": 8}])
    with pytest.raises(ConfigurationError, match="selector conflicts with row tp"):
        aic_fpm_database.prepare_aic_fpm_database({}, artifact)
    assert not overlays


def test_missing_row_contract_and_corrupt_parquet_are_explicit_failures(source):
    write, overlays = source
    artifact = write(rows=[{key: value for key, value in ROW.items() if key != "measurement_policy"}])
    with pytest.raises(ConfigurationError, match="contract columns"):
        aic_fpm_database.prepare_aic_fpm_database({}, artifact)
    artifact.local_path.write_bytes(b"not parquet")
    artifact.sha256 = digest(artifact.local_path)
    metadata = json.loads(artifact.local_metadata_path.read_text())
    metadata["parquet_sha256"] = artifact.sha256
    artifact.local_metadata_path.write_text(json.dumps(metadata))
    artifact.metadata_sha256 = digest(artifact.local_metadata_path)
    with pytest.raises(ConfigurationError, match="Cannot validate reconstructed"):
        aic_fpm_database.prepare_aic_fpm_database({}, artifact)
    assert not overlays


def test_unmarked_missing_metadata_is_not_guessed(source):
    write, _ = source
    artifact = write(omit=("metadata_provenance",))
    prepared = aic_fpm_database.prepare_aic_fpm_database({}, artifact)
    try:
        staged = next(prepared.systems_root.rglob("fpm_forward_perf.metadata.json"))
        assert staged.is_symlink()
        assert "coordinate_system" not in json.loads(staged.read_text())
        assert "fpm_metadata_recovery" not in prepared.diagnostics
    finally:
        prepared.close()


def test_mixed_source_binds_staging_provenance_without_changing_source_identity(source, tmp_path):
    from scripts.fpm_accuracy.models.learned_scenario import _bounded_native_diagnostics
    from scripts.fpm_accuracy.models.mixed_scenario import MixedOriginalA

    write, _ = source
    artifact = write()
    mixed = MixedOriginalA({}, artifact, "ac")
    original_hashes = (artifact.sha256, artifact.metadata_sha256)
    manifests = []

    def manifest_writer(path, rows, worker_type, **kwargs):
        path.write_text(json.dumps(kwargs["provenance"]))
        manifests.append(path)
        return {"manifest_path": str(path), "source_sha256": digest(path)}

    native = SimpleNamespace(
        normalize_config=lambda config: config,
        estimate_forward_pass_with_route=lambda rows: None,
        mixed_composition_diagnostics=lambda: {},
    )
    try:
        staging = mixed.provenance["staging"]
        assert staging == mixed.prepared.diagnostics
        recovery = staging["fpm_metadata_recovery"]
        assert recovery["source_metadata_sha256"] == artifact.metadata_sha256
        assert recovery["recovered_fields"]["coordinate_system"] == "iteration_totals_balanced_v1"
        assert mixed.provenance["grouping"]["coordinate_system"] is None  # Original source remains explicit.
        mixed.prepared.diagnostics["fpm_metadata_recovery"]["recovered_fields"]["system"] = "changed"
        assert recovery["recovered_fields"]["system"] == "h200_sxm"
        mixed.bind(
            native,
            SimpleNamespace(write_original_a_manifest=manifest_writer),
            {"worker_type": "aggregated", "estimator_config": {}},
            [],
            tmp_path / "seed.json",
            20,
            tmp_path,
        )
        recorded = json.loads(manifests[0].read_text())
        assert recorded["staging"] == staging
        assert (recorded["source_sha256"], recorded["metadata_sha256"]) == original_hashes
        assert [item["sha256"] for item in recorded["sources"]] == list(original_hashes)
        public = _bounded_native_diagnostics(recorded)
        assert public["staging"] == staging
        assert all(item["path"] == "[local path omitted]" for item in public["sources"])
        assert str(tmp_path) not in json.dumps(public)
        assert (digest(artifact.local_path), digest(artifact.local_metadata_path)) == original_hashes
    finally:
        mixed.close()
