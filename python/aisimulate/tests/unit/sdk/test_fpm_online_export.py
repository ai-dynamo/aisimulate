# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import math
from types import SimpleNamespace

import pytest

from aisimulate_core.sdk import fpm_learned

from . import fpm_reference
from .fpm_training_support import import_sklearn
from .fpm_training_support import isolate_training_dependencies as isolate_training_dependencies

pytestmark = pytest.mark.unit

np = pytest.importorskip("numpy")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
sklearn = import_sklearn()


@pytest.fixture
def identity(monkeypatch):
    """Stub only Rust identity resolution; fitting/export use real sklearn."""
    import aisimulate_core

    resolved = {
        "model": "test-model",
        "system": "h100_sxm",
        "backend": "vllm",
        "backend_version": "0.10.0",
        "worker_type": "decode",
        "tp": 2,
        "pp": 1,
        "attention_dp": 1,
        "dcp": 1,
        "moe_tp_size": 2,
        "moe_ep_size": 1,
        "gemm_quant_mode": "fp8",
        "kvcache_quant_mode": "fp8",
        "fpm_profile": {"model": {"num_layers": 4}, "resources": {"gpu_memory_gb": 80}},
    }

    def normalize(payload):
        assert json.loads(payload) == resolved
        return json.dumps(resolved)

    monkeypatch.setitem(
        aisimulate_core.__dict__, "RustForwardPassPerfModel", SimpleNamespace(learned_identity=normalize)
    )
    return resolved


def _iterations():
    # The first two rows are rejected, so exported IDs cannot be post-filter
    # positions. Every adjacent pair shares X but has distinct observation IDs.
    rows = [[{"version": 1, "wall_time": 0.0, "scheduled_requests": {}}]]
    rows.append(
        [{"version": 1, "wall_time": 0.01, "scheduled_requests": {"num_prefill_requests": 1, "sum_prefill_tokens": 2}}]
    )
    for index in range(64):
        count = 1 + index // 2
        rows.append(
            [
                {
                    "version": 1,
                    "worker_id": "observed-worker",
                    "dp_rank": 0,
                    "counter_id": 1000 + index,
                    "wall_time": (1 + count * 0.2 + index % 2 * 0.01) / 1000,
                    "scheduled_requests": {"num_decode_requests": count, "sum_decode_kv_tokens": count * 100},
                }
            ]
        )
    return rows


def _train_kwargs():
    return {
        "features": ("num_decode_requests", "sum_decode_kv_tokens"),
        "max_iter": 4,
        "max_leaf_nodes": 3,
        "min_samples_leaf": 2,
        "min_store_rows": 5,
        "learning_rate": 0.2,
        "early_stopping": False,
    }


@pytest.mark.parametrize("target", ["ms", "log_ms"])
def test_checkpoint_preserves_fitted_rows_mapper_and_prediction_artifact(tmp_path, monkeypatch, identity, target):
    models = []
    original_export = fpm_learned._export_hgb

    def capture(model):
        models.append(model)
        return original_export(model)

    monkeypatch.setattr(fpm_learned, "_export_hgb", capture)

    class Reiterable:
        # Telemetry campaigns stream disk-backed input; export must not require
        # len(iterations), indexing, or retaining every raw telemetry dictionary.
        def __iter__(self):
            return iter(_iterations())

    destination = tmp_path / "checkpoint"
    original = fpm_learned.train(Reiterable(), "decode", target=target, identity=identity, **_train_kwargs())
    artifact = fpm_learned.train(
        Reiterable(), "decode", target=target, online_checkpoint_dir=destination, identity=identity, **_train_kwargs()
    )
    assert artifact == original
    assert json.loads((destination / "model.json").read_bytes()) == original
    manifest = json.loads((destination / "manifest.json").read_bytes())
    assert manifest["schema"] == "aisimulate_fpm_training_checkpoint"
    assert manifest["schema_version"] == 1
    assert manifest["shrinkage"] == "included_in_leaf_values"
    assert manifest["sklearn_version"] == sklearn.__version__
    assert manifest["features"] == artifact["features"]
    assert manifest["target"] == target
    assert manifest["identity"] == identity
    assert manifest["identity"]["fpm_profile"] == identity["fpm_profile"]
    assert manifest["artifact_sha256"] == hashlib.sha256((destination / "model.json").read_bytes()).hexdigest()
    store = manifest["stores"]["pure_decode"]
    parquet = destination / store["rows_file"]
    assert store["rows_sha256"] == hashlib.sha256(parquet.read_bytes()).hexdigest()
    table = pq.read_table(parquet)
    assert table["row_id"].type == pa.uint64()
    assert table["y"].type == table["wall_time_ms"].type == table["x0"].type == pa.float64()
    assert table["row_id"].to_pylist() == list(range(2, 66))
    assert table["x0"][0].as_py() == table["x0"][1].as_py()
    assert table["row_id"][0].as_py() != table["row_id"][1].as_py()
    assert json.loads(table["source_json"][0].as_py())[0]["counter_id"] == 1000
    wall = [row[0]["wall_time"] * 1000.0 for row in _iterations()[2:]]
    assert table["wall_time_ms"].to_pylist() == wall
    assert table["y"].to_pylist() == ([math.log(value) for value in wall] if target == "log_ms" else wall)
    assert store["row_count"] == len(wall) == artifact["metadata"]["train_rows"]["pure_decode"]
    fitted = models[-1]
    assert store["loss"] == "squared_error"
    assert store["learning_rate"] == fitted.learning_rate
    assert store["l2_regularization"] == fitted.l2_regularization
    assert store["min_samples_leaf"] == fitted.min_samples_leaf
    assert store["max_depth"] == fitted.max_depth
    assert store["max_leaf_nodes"] == fitted.max_leaf_nodes
    assert store["bin_thresholds"] == [values.tolist() for values in fitted._bin_mapper.bin_thresholds_]
    assert store["n_bins_non_missing"] == fitted._bin_mapper.n_bins_non_missing_.tolist()
    x = np.column_stack([table[f"x{column}"].to_numpy() for column in range(2)])
    sklearn_values = fitted.predict(x)
    model = artifact["stores"]["pure_decode"]
    exported_values = [
        model["baseline"] + sum(fpm_reference._tree_leaf_value(tree, row) for tree in model["trees"]) for row in x
    ]
    np.testing.assert_allclose(exported_values, sklearn_values, rtol=1e-14, atol=1e-14)


def test_checkpoint_bins_preserve_missing_values_and_boundary_routing(tmp_path):
    from sklearn.ensemble import HistGradientBoostingRegressor

    x = np.arange(80, dtype=np.float64).reshape(-1, 1)
    x[::5] = np.nan
    wall = np.nan_to_num(x[:, 0], nan=40.0) + 1.0
    fitted = HistGradientBoostingRegressor(
        early_stopping=False, max_iter=1, max_leaf_nodes=2, min_samples_leaf=5, max_bins=8, random_state=0
    ).fit(x, wall)
    metadata = fpm_learned._online_store_metadata(fitted, 1)
    artifact = {
        "schema": fpm_learned.SCHEMA_NAME,
        "schema_version": 1,
        "worker_type": "decode",
        "features": ["sum_decode_kv_tokens"],
        "target": "ms",
        "stores": {"pure_decode": fpm_learned._export_hgb(fitted)},
        "metadata": {},
    }
    destination = tmp_path / "missing"
    fpm_learned._write_training_checkpoint(
        destination,
        artifact,
        {"worker_type": "decode"},
        {"pure_decode": (x.tolist(), wall.tolist())},
        {"pure_decode": [(index, "[]") for index in range(len(x))]},
        {"pure_decode": metadata},
    )
    table = pq.read_table(destination / "pure_decode.rows.parquet")
    assert table["x0"].null_count == 0  # IEEE NaN must not become Arrow null.
    np.testing.assert_equal(table["x0"].to_numpy(), x[:, 0])
    boundaries = np.array(metadata["bin_thresholds"][0])
    probes = np.concatenate(([-100.0, 1000.0, np.nan], boundaries, np.nextafter(boundaries, np.inf)))
    expected_bins = fitted._bin_mapper.transform(probes.reshape(-1, 1))[:, 0]
    # Independent interoperability check against the fitted sklearn bin mapper:
    # numeric thresholds include equality on the left, NaN has its own bin.
    exported_bins = np.searchsorted(boundaries, probes, side="left")
    exported_bins[np.isnan(probes)] = metadata["missing_values_bin_idx"]
    np.testing.assert_array_equal(exported_bins, expected_bins)


@pytest.mark.parametrize("field", ["trainer", "model", "hyperparameters", "train_rows", "trees_fitted"])
def test_checkpoint_rejects_metadata_that_overrides_training_truth(tmp_path, field):
    with pytest.raises(ValueError, match="cannot override training fields"):
        fpm_learned.train(
            _iterations(),
            "decode",
            online_checkpoint_dir=tmp_path / "bad",
            metadata={field: "forged"},
            **_train_kwargs(),
        )
    assert not (tmp_path / "bad").exists()


def test_checkpoint_rejects_early_stopping_and_requires_identity(tmp_path):
    kwargs = _train_kwargs() | {"early_stopping": True}
    with pytest.raises(ValueError, match="early_stopping=False"):
        fpm_learned.train(_iterations(), "decode", online_checkpoint_dir=tmp_path / "bad", **kwargs)
    with pytest.raises(ValueError, match="requires an identity"):
        fpm_learned.train(_iterations(), "decode", online_checkpoint_dir=tmp_path / "bad", **_train_kwargs())
    assert not (tmp_path / "bad").exists()


def test_checkpoint_rejects_worker_identity_mismatch(monkeypatch):
    import aisimulate_core

    monkeypatch.setitem(
        aisimulate_core.__dict__,
        "RustForwardPassPerfModel",
        SimpleNamespace(learned_identity=lambda _: json.dumps({"worker_type": "prefill"})),
    )
    with pytest.raises(ValueError, match="worker_type must match"):
        fpm_learned._checkpoint_identity({"worker_type": "prefill"}, "decode")


def test_checkpoint_version_guard_does_not_restrict_prediction_artifact_training(tmp_path, monkeypatch):
    monkeypatch.setattr(sklearn, "__version__", "0.unsupported")
    with pytest.raises(ValueError, match="does not support sklearn"):
        fpm_learned.train(_iterations(), "decode", online_checkpoint_dir=tmp_path / "bad", **_train_kwargs())
    assert fpm_learned.train(_iterations(), "decode", **_train_kwargs())["stores"]


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("loss", "absolute_error", "squared_error"),
        ("monotonic_cst", [1], "constraints"),
        ("interaction_cst", [[0]], "constraints"),
        ("max_features", 0.5, "feature subsampling"),
    ],
)
def test_checkpoint_rejects_unsupported_fit_semantics(parameter, value, message):
    from sklearn.ensemble import HistGradientBoostingRegressor

    fitted = HistGradientBoostingRegressor(
        max_iter=1, early_stopping=False, min_samples_leaf=2, **{parameter: value}
    ).fit(np.arange(40).reshape(-1, 1), np.arange(40))
    with pytest.raises(ValueError, match=message):
        fpm_learned._online_store_metadata(fitted, 1)


def test_checkpoint_hash_verification_and_atomic_failure(tmp_path, monkeypatch, identity):
    destination = tmp_path / "checkpoint"
    fpm_learned.train(_iterations(), "decode", online_checkpoint_dir=destination, identity=identity, **_train_kwargs())
    manifest = json.loads((destination / "manifest.json").read_bytes())
    fpm_learned._verify_checkpoint_files(destination, manifest)
    parquet = destination / manifest["stores"]["pure_decode"]["rows_file"]
    parquet.write_bytes(parquet.read_bytes() + b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        fpm_learned._verify_checkpoint_files(destination, manifest)
    original = (destination / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        fpm_learned.train(
            _iterations(), "decode", online_checkpoint_dir=destination, identity=identity, **_train_kwargs()
        )
    assert (destination / "manifest.json").read_bytes() == original

    def reject_before_publication(staging, manifest):
        assert (staging / "model.json").exists()
        assert not (staging / "manifest.json").exists()
        raise ValueError("injected integrity failure")

    monkeypatch.setattr(fpm_learned, "_verify_checkpoint_files", reject_before_publication)
    with pytest.raises(ValueError, match="integrity failure"):
        fpm_learned.train(
            _iterations(), "decode", online_checkpoint_dir=tmp_path / "failed", identity=identity, **_train_kwargs()
        )
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".failed.tmp-*"))


@pytest.mark.parametrize("bad_field", ["features", "wall_time_ms", "row_ids", "bin_thresholds"])
def test_checkpoint_rejects_inconsistent_training_data_without_publication(tmp_path, bad_field):
    from sklearn.ensemble import HistGradientBoostingRegressor

    x = np.arange(20, dtype=np.float64).reshape(-1, 1)
    wall = list(range(1, 21))
    fitted = HistGradientBoostingRegressor(early_stopping=False, max_iter=1, min_samples_leaf=2).fit(x, wall)
    if bad_field == "bin_thresholds":
        fitted._bin_mapper.bin_thresholds_[0][0] = math.inf
        with pytest.raises(ValueError, match="invalid numeric thresholds"):
            fpm_learned._online_store_metadata(fitted, 1)
        return
    metadata = fpm_learned._online_store_metadata(fitted, 1)
    artifact = {
        "schema": fpm_learned.SCHEMA_NAME,
        "schema_version": 1,
        "worker_type": "decode",
        "features": ["num_decode_requests"],
        "target": "ms",
        "stores": {"pure_decode": fpm_learned._export_hgb(fitted)},
        "metadata": {},
    }
    source = [(index, "[]") for index in range(len(x))]
    if bad_field == "features":
        x[0, 0] = math.inf
    elif bad_field == "wall_time_ms":
        wall[0] = 0.0
    else:
        source[1] = source[0]
    destination = tmp_path / "invalid"
    with pytest.raises(ValueError, match="training features|wall_time_ms|unique IDs"):
        fpm_learned._write_training_checkpoint(
            destination,
            artifact,
            {"worker_type": "decode"},
            {"pure_decode": (x.tolist(), wall)},
            {"pure_decode": source},
            {"pure_decode": metadata},
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".invalid.tmp-*"))


def test_cli_writes_checkpoint_and_the_same_prediction_artifact(tmp_path, monkeypatch, identity):
    records = tmp_path / "measurements.jsonl"
    records.write_text("\n".join(json.dumps(row[0]) for row in _iterations()))
    config = tmp_path / "identity.json"
    config.write_text(json.dumps(identity))
    checkpoint = tmp_path / "checkpoint"
    prediction = tmp_path / "prediction.json"
    # Evaluation is a separate compiled-Rust contract; exercise real training
    # and publication while keeping this CLI/export test independent of the FFI.
    monkeypatch.setattr(fpm_learned, "evaluate", lambda artifact, rows: {})
    result = fpm_learned.main(
        [
            "train",
            "--fpm",
            str(records),
            "--worker-type",
            "decode",
            "--out",
            str(prediction),
            "--online-checkpoint-dir",
            str(checkpoint),
            "--identity",
            str(config),
            "--features",
            "num_decode_requests,sum_decode_kv_tokens",
            "--max-iter",
            "2",
            "--holdout-frac",
            "0",
        ]
    )
    assert result == 0
    assert prediction.read_bytes() == (checkpoint / "model.json").read_bytes()
    assert (checkpoint / "manifest.json").is_file()


def test_training_identity_is_bound_in_frozen_artifact_and_conflicts_fail(identity):
    artifact = fpm_learned.train(_iterations(), "decode", identity=identity, **_train_kwargs())
    assert artifact["metadata"]["identity"] == identity
    with pytest.raises(ValueError, match="metadata.identity conflicts"):
        fpm_learned.train(
            _iterations(), "decode", identity=identity, metadata={"identity": {"model": "different"}}, **_train_kwargs()
        )


def test_concurrent_checkpoint_destination_creation_is_not_overwritten(tmp_path, monkeypatch, identity):
    destination = tmp_path / "checkpoint"
    original_verify = fpm_learned._verify_checkpoint_files

    def competing_export(directory, manifest):
        original_verify(directory, manifest)
        destination.mkdir()
        (destination / "owned-by-other-writer").write_text("preserve")

    monkeypatch.setattr(fpm_learned, "_verify_checkpoint_files", competing_export)
    with pytest.raises(FileExistsError):
        fpm_learned.train(
            _iterations(), "decode", identity=identity, online_checkpoint_dir=destination, **_train_kwargs()
        )
    assert (destination / "owned-by-other-writer").read_text() == "preserve"
    assert sorted(path.name for path in destination.iterdir()) == ["owned-by-other-writer"]
