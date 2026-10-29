# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original-A exporter preserves eligible groups and immutable seed references."""

import hashlib
import json

import pytest

from aisimulate_core.sdk import fpm_learned

pytestmark = pytest.mark.unit


def test_manifest_preserves_subthreshold_mixed_groups_and_zero_rank_label(tmp_path, monkeypatch):
    identity = {"model": "model", "worker_type": "aggregated", "attention_dp": 2}
    monkeypatch.setattr(fpm_learned, "_checkpoint_identity", lambda config, role: identity)

    def group(counter, *, mixed=False, wall=0.01):
        return [
            {
                "version": 1,
                "worker_id": "worker",
                "counter_id": counter,
                "dp_rank": rank,
                "wall_time": wall if rank == 0 else 0.0,
                "scheduled_requests": {
                    "num_decode_requests": 1,
                    "sum_decode_kv_tokens": 64,
                    "num_prefill_requests": int(mixed),
                    "sum_prefill_tokens": 8 * int(mixed),
                },
            }
            for rank in range(2)
        ]

    groups = [group(0), group(1, mixed=True), group(2, wall=0.0)]
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps({"features": fpm_learned.FEATURE_PRESETS["fpm5"]}))
    destination = tmp_path / "original-a.json"
    kwargs = dict(identity=identity, min_store_rows=20, seed_artifact_paths=[seed], provenance={"grouping": {}})
    source = fpm_learned.write_original_a_manifest(destination, groups, "aggregated", **kwargs)
    content = destination.read_bytes()
    manifest = json.loads(content)
    assert manifest["accepted_iterations"] == groups[:2]
    assert manifest["min_store_rows"] == 20
    assert manifest["identity"] == identity
    assert manifest["seed_artifacts"] == [
        {"path": str(seed.resolve()), "sha256": hashlib.sha256(seed.read_bytes()).hexdigest()}
    ]
    assert source == {"manifest_path": str(destination), "source_sha256": hashlib.sha256(content).hexdigest()}
    groups[0][0]["wall_time"] = 99.0
    assert destination.read_bytes() == content
    with pytest.raises(FileExistsError):
        fpm_learned.write_original_a_manifest(destination, groups, "aggregated", **kwargs)


@pytest.mark.parametrize("threshold", [True, 0, -1, 1.5])
def test_original_a_rejects_invalid_training_threshold(tmp_path, threshold):
    with pytest.raises(ValueError, match="positive integer"):
        fpm_learned.write_original_a_manifest(
            tmp_path / "a.json",
            [],
            "aggregated",
            identity={},
            min_store_rows=threshold,
            seed_artifact_paths=[],
            provenance={},
        )


def test_original_a_records_training_checkpoint_anchor(tmp_path, monkeypatch):
    monkeypatch.setattr(fpm_learned, "_checkpoint_identity", lambda config, role: {})
    group = [{"version": 1, "wall_time": 0.01, "scheduled_requests": {"num_decode_requests": 1}}]
    seed = tmp_path / "model.json"
    seed.write_text(json.dumps({"features": fpm_learned.FEATURE_PRESETS["fpm5"]}))
    checkpoint = tmp_path / "manifest.json"
    checkpoint.write_text('{"schema":"aisimulate_fpm_training_checkpoint"}')
    output = tmp_path / "original-a.json"
    fpm_learned.write_original_a_manifest(
        output, [group], "aggregated", identity={}, min_store_rows=20, seed_artifact_paths=[seed], provenance={}
    )
    reference = json.loads(output.read_text())["seed_artifacts"][0]
    assert reference["checkpoint_manifest_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def test_streamed_manifest_matches_legacy_canonical_bytes(tmp_path, monkeypatch):
    identity = {"worker_type": "aggregated", "nested": {"z": [None, True, -0.0], "a": '模型\n\\"'}}
    monkeypatch.setattr(fpm_learned, "_checkpoint_identity", lambda config, role: identity)
    groups = [
        [
            {
                "version": 1,
                "worker_id": "worker",
                "counter_id": index,
                "wall_time": wall,
                "scheduled_requests": {"num_decode_requests": 1, "sum_decode_kv_tokens": index * 17},
                "nested": {"z": [index, None, False], "a": {"escaped": "\tµ"}},
            }
        ]
        for index, wall in enumerate((0.00125, 0.0, 0.02, 0.5))
    ]
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps({"features": fpm_learned.FEATURE_PRESETS["fpm5"]}))
    provenance = {"z": [2.5, {"b": False, "a": "é"}], "grouping": {"balanced": True}}
    expected = {
        "schema_version": 1,
        "source_kind": "self_benchmark",
        "identity": identity,
        "min_store_rows": 20,
        "accepted_iterations": [groups[index] for index in (0, 2, 3)],
        "seed_artifacts": [{"path": str(seed.resolve()), "sha256": hashlib.sha256(seed.read_bytes()).hexdigest()}],
        "provenance": provenance,
    }
    legacy = json.dumps(expected, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    destination = tmp_path / "original-a.json"
    source = fpm_learned.write_original_a_manifest(
        destination,
        groups,
        "aggregated",
        identity=identity,
        min_store_rows=20,
        seed_artifact_paths=[seed],
        provenance=provenance,
    )
    assert destination.read_bytes() == legacy
    assert source["source_sha256"] == hashlib.sha256(legacy).hexdigest()


def test_streamed_manifest_failure_removes_only_its_partial_output(tmp_path, monkeypatch):
    monkeypatch.setattr(fpm_learned, "_checkpoint_identity", lambda config, role: {})
    group = [{"version": 1, "wall_time": 0.01, "scheduled_requests": {"num_decode_requests": 1}}]
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps({"features": fpm_learned.FEATURE_PRESETS["fpm5"]}))
    destination = tmp_path / "original-a.json"
    kwargs = dict(identity={}, min_store_rows=20, seed_artifact_paths=[seed], provenance={"invalid": float("nan")})
    with pytest.raises(ValueError):
        fpm_learned.write_original_a_manifest(destination, [group], "aggregated", **kwargs)
    assert not destination.exists()
    destination.write_bytes(b"existing manifest")
    with pytest.raises(FileExistsError):
        fpm_learned.write_original_a_manifest(destination, [group], "aggregated", **kwargs)
    assert destination.read_bytes() == b"existing manifest"


@pytest.mark.parametrize(
    "seed_content",
    [
        [],
        {},
        {"features": []},
        {"features": "num_decode_requests"},
        {"features": ["unknown_feature"]},
        {"features": [None]},
        {"features": ["num_decode_requests", "num_decode_requests"]},
    ],
)
def test_original_a_rejects_invalid_seed_features_without_creating_output(tmp_path, monkeypatch, seed_content):
    monkeypatch.setattr(fpm_learned, "_checkpoint_identity", lambda config, role: {})
    group = [{"version": 1, "wall_time": 0.01, "scheduled_requests": {"num_decode_requests": 1}}]
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps(seed_content))
    destination = tmp_path / "original-a.json"
    with pytest.raises(ValueError, match="features"):
        fpm_learned.write_original_a_manifest(
            destination,
            [group],
            "aggregated",
            identity={},
            min_store_rows=20,
            seed_artifact_paths=[seed],
            provenance={},
        )
    assert not destination.exists()


@pytest.mark.parametrize(
    "second_features",
    [fpm_learned.FEATURE_PRESETS["atomic10"], tuple(reversed(fpm_learned.FEATURE_PRESETS["fpm5"]))],
)
def test_original_a_rejects_different_seed_feature_lists(tmp_path, monkeypatch, second_features):
    monkeypatch.setattr(fpm_learned, "_checkpoint_identity", lambda config, role: {})
    group = [{"version": 1, "wall_time": 0.01, "scheduled_requests": {"num_decode_requests": 1}}]
    seeds = [tmp_path / "seed-1.json", tmp_path / "seed-2.json"]
    for seed, features in zip(seeds, [fpm_learned.FEATURE_PRESETS["fpm5"], second_features], strict=True):
        seed.write_text(json.dumps({"features": features}))
    destination = tmp_path / "original-a.json"
    with pytest.raises(ValueError, match="same ordered features"):
        fpm_learned.write_original_a_manifest(
            destination, [group], "aggregated", identity={}, min_store_rows=20, seed_artifact_paths=seeds, provenance={}
        )
    assert not destination.exists()


def test_original_a_accepts_seeds_with_matching_features(tmp_path, monkeypatch):
    monkeypatch.setattr(fpm_learned, "_checkpoint_identity", lambda config, role: {})
    group = [{"version": 1, "wall_time": 0.01, "scheduled_requests": {"num_decode_requests": 1}}]
    seeds = [tmp_path / "seed-1.json", tmp_path / "seed-2.json"]
    for seed in seeds:
        seed.write_text(json.dumps({"features": fpm_learned.FEATURE_PRESETS["fpm5"]}))
    destination = tmp_path / "original-a.json"
    fpm_learned.write_original_a_manifest(
        destination, [group], "aggregated", identity={}, min_store_rows=20, seed_artifact_paths=seeds, provenance={}
    )
    manifest = json.loads(destination.read_text())
    assert manifest["accepted_iterations"] == [group]
    assert [seed["path"] for seed in manifest["seed_artifacts"]] == [str(path.resolve()) for path in seeds]
