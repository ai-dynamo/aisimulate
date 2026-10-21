# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real sklearn/Parquet/native interoperability, without exporter or API mocks."""

from __future__ import annotations

import copy
import json
import math

import pytest

from aisimulate_core.sdk import fpm_learned
from aisimulate_core.sdk.rust_engine_step import ForwardPassPerfModelConfig, RustForwardPassPerfModel

from .fpm_reference import reference_predict_ms
from .fpm_training_support import import_sklearn
from .fpm_training_support import isolate_training_dependencies as isolate_training_dependencies

pytestmark = pytest.mark.unit
import_sklearn()
pytest.importorskip("pyarrow")

IDENTITY = {
    "model": "offline/model",
    "system": "offline_system",
    "backend": "vllm",
    "worker_type": "decode",
    "estimation_mode": "fpm_regression",
}


def iteration(index, *, shifted=False):
    count = 1 + index % 16
    requests = {"num_decode_requests": count, "sum_decode_kv_tokens": count * (100 + index % 3)}
    # A synchronized two-rank iteration, including optional u64 request lists.
    # Its label is the second rank's latency, not the first's or their sum.
    requests["extend_lengths"] = [1] * count
    requests["past_kv_lengths"] = [100 + index % 3] * count
    milliseconds = 1.3 + count * 0.17 + index % 3 * 0.03 + (1.7 if shifted else 0)
    return [
        {"version": 1, "dp_rank": 0, "wall_time": milliseconds / 2000, "scheduled_requests": requests},
        {"version": 1, "dp_rank": 1, "wall_time": milliseconds / 1000, "scheduled_requests": requests},
    ]


def train(tmp_path, target="log_ms", *, identity=IDENTITY):
    rows = [iteration(index) for index in range(64)]
    path = tmp_path / "seed"
    artifact = fpm_learned.train(
        rows,
        "decode",
        features=("num_decode_requests", "sum_decode_kv_tokens"),
        target=target,
        max_iter=4,
        max_leaf_nodes=3,
        min_samples_leaf=2,
        min_store_rows=5,
        learning_rate=0.2,
        online_checkpoint_dir=path,
        identity=identity,
    )
    return rows, path, artifact


def load(path, mode, refresh="eager", *, identity=IDENTITY):
    return RustForwardPassPerfModel.best_available(
        {
            **identity,
            "estimation_mode": "fpm_learned",
            "estimator_config": {
                "correction": {"enabled": False},
                "fpm_learned": {
                    "checkpoint_path": str(path),
                    "adaptation": "frozen" if mode == "frozen" else "online",
                    "update_mode": "structural" if mode == "frozen" else mode,
                    "derivative_refresh": refresh,
                    "recent_window": 5,
                    "update_every": 3,
                    "eager_refresh_interval": 4,
                },
            },
        }
    )


def test_native_identity_projection_preserves_export_and_load_compatibility(tmp_path):
    deployment = {
        **IDENTITY,
        "backend_version": "0.10.0",
        "tp": 2,
        "dcp": 1,
        "gemm_quant_mode": "fp8",
        "kv_block_size": 16,
    }
    # These training-time choices do not bind the deployment. Loading below
    # changes estimator, correction, source roots, and selection policies.
    training_data = tmp_path / "training-data"
    training_data.mkdir()
    training = {
        **deployment,
        "fallback_policy": "allow",
        "database_mode": "HYBRID",
        "transfer_policy": ["xshape"],
        "systems_paths": [str(training_data)],
        "strict_provenance": True,
        "estimator_config": {"correction": {"enabled": True}},
    }
    expected = RustForwardPassPerfModel.learned_identity(ForwardPassPerfModelConfig(**deployment))
    assert RustForwardPassPerfModel.learned_identity(training) == expected
    assert expected["pp"] == expected["attention_dp"] == 1
    assert {key: expected[key] for key in deployment if key != "estimation_mode"} == {
        key: value for key, value in deployment.items() if key != "estimation_mode"
    }

    rows, checkpoint, artifact = train(tmp_path, identity=training)
    assert artifact["metadata"]["identity"] == expected
    assert json.loads((checkpoint / "manifest.json").read_text())["identity"] == expected
    for mode in ("frozen", "leaf_only"):
        model = load(checkpoint, mode, identity=deployment)
        assert model.export_learned_artifact() == artifact
        assert model.learned_online_diagnostics()["identity_verified"]
        assert model.estimate_forward_pass_time_ms(rows[0]) == pytest.approx(reference_predict_ms(artifact, rows[0]))

    # Both frozen and adaptive checkpoint loaders reject an actual deployment
    # change even when the exported forest could still make a prediction.
    for change in (
        {"model": "different/model"},
        {"system": "different_system"},
        {"backend_version": "0.11.0"},
        {"worker_type": "prefill"},
        {"tp": 4},
        {"dcp": 2},
        {"gemm_quant_mode": "bfloat16"},
        {"kv_block_size": 32},
    ):
        for mode in ("frozen", "leaf_only"):
            with pytest.raises(ValueError, match="identity does not match"):
                load(checkpoint, mode, identity={**deployment, **change})


def leaf_refit(artifact, rows):
    """Slow fresh recomputation using row routing, without native cached state."""
    result = copy.deepcopy(artifact)
    named = [fpm_learned.compute_features(row) for row in rows]
    features = [[values[key] for key in artifact["features"]] for values in named]
    targets = [fpm_learned.iteration_wall_ms(row) for row in rows]
    if artifact["target"] == "log_ms":
        targets = [math.log(value) for value in targets]
    store = result["stores"]["pure_decode"]
    prefix = [store["baseline"]] * len(rows)
    for tree in store["trees"]:
        memberships = {}
        for row_index, values in enumerate(features):
            node = 0
            while tree["left"][node] >= 0:
                value = values[tree["feature"][node]]
                left = tree["missing_left"][node] if math.isnan(value) else value <= tree["threshold"][node]
                node = tree["left" if left else "right"][node]
            memberships.setdefault(node, []).append(row_index)
        for node, indexes in memberships.items():
            value = 0.2 * sum(targets[index] - prefix[index] for index in indexes) / len(indexes)
            tree["value"][node] = value
            for index in indexes:
                prefix[index] += value
    return result


@pytest.mark.parametrize("target", ["ms", "log_ms"])
def test_leaf_only_matches_slow_recomputation_after_real_export_and_fifo(tmp_path, target):
    anchors, path, artifact = train(tmp_path, target)
    model = load(path, "leaf_only")
    for row in anchors:
        assert model.estimate_forward_pass_time_ms(row) == pytest.approx(reference_predict_ms(artifact, row), rel=1e-14)
    recent = []
    expected = artifact
    for batch in range(5):
        incoming = [iteration(batch * 3 + index, shifted=True) for index in range(3)]
        recent = (recent + incoming)[-5:]
        model.tune_with_fpms(incoming)
        expected = leaf_refit(expected, anchors + recent)
        for row in anchors[:16]:
            assert model.estimate_forward_pass_time_ms(row) == pytest.approx(
                reference_predict_ms(expected, row), rel=1e-12, abs=1e-12
            )
    counts = model.learned_online_diagnostics()["stores"]["pure_decode"]
    assert counts["anchor_rows"] == 64
    assert counts["accepted_observations"] == 15  # Neither ranks nor API calls.
    assert counts["recent_rows"] == 5
    assert counts["evicted_observations"] == 10


@pytest.mark.parametrize("mode", ["leaf_only", "structural"])
@pytest.mark.parametrize("refresh", ["eager", "lazy"])
def test_real_checkpoint_continuation_and_frozen_snapshot(tmp_path, mode, refresh):
    anchors, path, original = train(tmp_path)
    model = load(path, mode, refresh)
    assert model.export_learned_artifact() == original
    model.tune_with_fpms([iteration(index, shifted=True) for index in range(7)])
    before = model.export_learned_artifact()
    diagnostics = model.learned_online_diagnostics()
    assert diagnostics["stores"]["pure_decode"]["pending_rows"] == 1
    checkpoint = tmp_path / "resume"
    model.save_online_checkpoint(checkpoint)
    assert model.export_learned_artifact() == before
    assert model.learned_online_diagnostics() == diagnostics
    resumed = load(checkpoint, mode, refresh)
    assert resumed.export_learned_artifact() == before
    # Allocator capacity can differ after deserialization, but behavior cannot.
    for index in range(7, 22):
        observation = iteration(index, shifted=True)
        assert resumed.estimate_forward_pass_time_ms(observation) == model.estimate_forward_pass_time_ms(observation)
        for current in (model, resumed):
            current.tune_with_fpms([observation])
        assert resumed.export_learned_artifact() == model.export_learned_artifact()
    model.flush_updates()
    resumed.flush_updates()
    assert resumed.export_learned_artifact() == model.export_learned_artifact()
    # Frozen checkpoints are usable even when their training-state bytes are unavailable.
    (checkpoint / "state.bin").unlink()
    frozen = load(checkpoint, "frozen")
    assert frozen.export_learned_artifact() == before
    for row in anchors[:16]:
        assert frozen.estimate_forward_pass_time_ms(row) == pytest.approx(reference_predict_ms(before, row), rel=1e-14)


def test_seed_checkpoint_rejects_changed_feature_contract_and_rows(tmp_path):
    _, path, _ = train(tmp_path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["features"].reverse()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="contract mismatch"):
        load(path, "leaf_only")
    manifest["features"].reverse()
    manifest_path.write_text(json.dumps(manifest))
    parquet = path / manifest["stores"]["pure_decode"]["rows_file"]
    with parquet.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load(path, "structural")


def test_sklearn_missing_only_split_imports_without_inventing_numeric_bins(tmp_path):
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor

    x = np.array([[0.0]] * 10 + [[np.nan]] * 10)
    y = np.array([1.0] * 10 + [10.0] * 10)
    fitted = HistGradientBoostingRegressor(
        early_stopping=False, max_iter=2, max_leaf_nodes=2, min_samples_leaf=2, random_state=0
    ).fit(x, y)
    assert fitted._bin_mapper.bin_thresholds_[0].size == 0
    assert math.isinf(fitted._predictors[0][0].nodes[0]["num_threshold"])
    artifact = {
        "schema": fpm_learned.SCHEMA_NAME,
        "schema_version": 1,
        "worker_type": "decode",
        "features": ["sum_decode_kv_tokens"],
        "target": "ms",
        "stores": {"pure_decode": fpm_learned._export_hgb(fitted)},
        "metadata": {"identity": fpm_learned._checkpoint_identity(IDENTITY, "decode")},
    }
    checkpoint = tmp_path / "missing-seed"
    fpm_learned._write_training_checkpoint(
        checkpoint,
        artifact,
        fpm_learned._checkpoint_identity(IDENTITY, "decode"),
        {"pure_decode": (x.tolist(), y.tolist())},
        {"pure_decode": [(index, "[]") for index in range(len(y))]},
        {"pure_decode": fpm_learned._online_store_metadata(fitted, 1)},
    )
    model = load(checkpoint, "structural")
    assert model.export_learned_artifact() == artifact
    row = {"wall_time": 0.002, "scheduled_requests": {"num_decode_requests": 1, "sum_decode_kv_tokens": 0}}
    assert model.estimate_forward_pass_time_ms(row) == fitted.predict([[0.0]])[0]
    model.tune_with_fpms([[row]] * 3)
    saved = tmp_path / "missing-resume"
    model.save_online_checkpoint(saved)
    resumed = load(saved, "structural")
    assert resumed.export_learned_artifact() == model.export_learned_artifact()
