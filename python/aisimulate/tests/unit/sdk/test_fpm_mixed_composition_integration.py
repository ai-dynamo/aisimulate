# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real mixed routing, source binding and checkpoint continuation through Python."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aisimulate_core.sdk import fpm_learned
from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

from . import test_fpm_phase_readiness as phase_fixtures
from .fpm_training_support import import_sklearn
from .fpm_training_support import isolate_training_dependencies as isolate_training_dependencies

phase_roots = phase_fixtures.phase_roots

pytestmark = pytest.mark.unit
import_sklearn()


def _iteration(kind="mixed", *, wall_ms=8.0, index=0):
    return [
        {
            "version": 1,
            "worker_id": "worker",
            "dp_rank": 0,
            "counter_id": index,
            "wall_time": wall_ms / 1000,
            "scheduled_requests": {
                "num_prefill_requests": int(kind != "decode"),
                "sum_prefill_tokens": int(kind != "decode"),
                "num_decode_requests": int(kind != "prefill"),
                "sum_decode_kv_tokens": int(kind != "prefill"),
            },
        }
    ]


@pytest.fixture
def mixed_seed(tmp_path, phase_roots):
    config, make_root = phase_roots
    root = Path(make_root("mixed-direct"))
    parquet = root / "data/h200_sxm/vllm/0.25.1/fpm_forward_perf.parquet"
    table = pq.read_table(parquet).to_pylist()
    baseline = deepcopy(next(row for row in table if row["workload_kind"] == "decode"))
    baseline.update(total_kv_read_tokens=0, latency_ms=1.0, cell_id="synthetic-decode-zero")
    table.append(baseline)
    pq.write_table(pa.Table.from_pylist(table), parquet)
    sidecar = parquet.with_suffix(".metadata.json")
    metadata = json.loads(sidecar.read_text())
    metadata.update(row_count=len(table), parquet_sha256=hashlib.sha256(parquet.read_bytes()).hexdigest())
    sidecar.write_text(json.dumps(metadata))
    config["systems_paths"] = [str(root)]
    rows = [
        _iteration(kind, wall_ms=2 if kind == "prefill" else 3, index=index)
        for kind in ("prefill", "decode")
        for index in range(20)
    ]
    seed = tmp_path / "seed"
    fpm_learned.train(
        rows,
        "aggregated",
        features=fpm_learned.FEATURE_PRESETS["fpm5"],
        max_iter=3,
        max_leaf_nodes=3,
        min_samples_leaf=2,
        min_store_rows=20,
        online_checkpoint_dir=seed,
        identity=config,
    )
    source = fpm_learned.write_original_a_manifest(
        tmp_path / "original-a.json",
        rows,
        "aggregated",
        identity=config,
        min_store_rows=20,
        seed_artifact_paths=[seed / "model.json"],
        provenance={
            "source_sha256": metadata["parquet_sha256"],
            "grouping": {"coordinate_system": "iteration_totals_balanced_v1", "synthetic_fixture": True},
            "sources": [
                {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in (parquet, sidecar)
            ],
        },
    )
    return config, seed, source


def _config(fixture, checkpoint=None, *, enabled=True, adaptation="online"):
    config, seed, source = fixture
    config = deepcopy(config)
    config["estimation_mode"] = "fpm_learned"
    estimator = config["estimator_config"]
    estimator["correction"] = {"enabled": adaptation == "causal_selector", "mode": "hybrid"}
    estimator["fpm_learned"] = {
        "checkpoint_path": str(checkpoint or seed),
        "adaptation": adaptation,
        "update_mode": "leaf_only",
        "recent_window": 8,
        "update_every": 3,
    }
    if enabled:
        estimator["temporary_mixed_composition"] = {
            "scenario": "a_c",
            "mode": "compose_calibrated",
            "original_a": source,
            "calibration": {"sampling": {"bins_per_axis": [2, 2], "max_observations": 8}, "min_observations": 2},
        }
    return config


def _load(fixture, checkpoint=None, **kwargs):
    return RustForwardPassPerfModel.best_available(_config(fixture, checkpoint, **kwargs))


def test_frozen_hybrid_calibrates_missing_mixed_store_without_training_it(mixed_seed):
    config = _config(mixed_seed, adaptation="frozen")
    estimator = config["estimator_config"]
    estimator["correction"] = {"enabled": True, "mode": "hybrid", "min_observations": 1}
    estimator["temporary_mixed_composition"]["calibration"]["min_observations"] = 1
    normalized = RustForwardPassPerfModel.normalize_config(config)
    assert RustForwardPassPerfModel.normalize_config(json.loads(json.dumps(normalized))) == normalized
    control_config = deepcopy(config)
    del control_config["estimator_config"]["temporary_mixed_composition"]
    model = RustForwardPassPerfModel.best_available(normalized)
    control = RustForwardPassPerfModel.best_available(control_config)
    try:
        frozen = model.export_learned_artifact()
        query = _iteration()
        assert control.estimate_forward_pass_time_ms(query) is None
        # Synthetic phase table: prefill 2 + decode 3 - decode-zero 1 = 4ms.
        assert model.estimate_forward_pass_time_ms(query) == 4.0
        for admitted in range(1, 31):
            model.tune_with_fpms([_iteration(wall_ms=8.0, index=admitted)])
            prediction = model.estimate_forward_pass_with_route(query)
            assert prediction["latency_ms"] == 8.0
            route = prediction["temporary_mixed_composition"]
            assert route["route"] == "interpolation"
            assert route["calibration_admissions"] == admitted
        assert model.export_learned_artifact() == frozen
        assert "contains_locally_mixed" not in frozen["stores"]
        with pytest.raises(ValueError, match="online GBDT or causal selector"):
            model.append_mixed_learned_checkpoint(mixed_seed[1])
        for kind in ("prefill", "decode"):
            query = _iteration(kind)
            assert model.estimate_forward_pass_time_ms(query) == control.estimate_forward_pass_time_ms(query)
            for current in (model, control):
                current.tune_with_fpms([_iteration(kind, wall_ms=6.0)])
            assert model.estimate_forward_pass_time_ms(query) == control.estimate_forward_pass_time_ms(query)
    finally:
        model.close()
        control.close()


@pytest.fixture
def c_augmented_artifact(mixed_seed, tmp_path):
    """Represent an already fitted C store without changing authenticated A."""
    config, seed, _ = mixed_seed
    c_rows = [_iteration(wall_ms=12.0, index=index) for index in range(20)]
    c_artifact = fpm_learned.train(
        c_rows,
        "aggregated",
        features=fpm_learned.FEATURE_PRESETS["fpm5"],
        max_iter=3,
        max_leaf_nodes=3,
        min_samples_leaf=2,
        min_store_rows=20,
        online_checkpoint_dir=tmp_path / "c-seed",
        identity=config,
    )
    assert set(c_artifact["stores"]) == {"contains_locally_mixed"}
    artifact = json.loads((seed / "model.json").read_text())
    assert artifact["metadata"]["identity"] == c_artifact["metadata"]["identity"]
    artifact["stores"].update(c_artifact["stores"])
    for field in ("train_rows", "trees_fitted"):
        artifact["metadata"].setdefault(field, {}).update(c_artifact["metadata"].get(field, {}))
    path = tmp_path / "a-plus-c-artifact.json"
    path.write_text(json.dumps(artifact, allow_nan=False))
    return path


def _augmented_config(mixed_seed, artifact, *, enabled=True):
    config = _config(mixed_seed, enabled=enabled, adaptation="frozen")
    config["estimator_config"]["fpm_learned"] = {
        "artifact_path": str(artifact),
        "adaptation": "frozen",
    }
    config["estimator_config"]["correction"] = {"enabled": True, "mode": "hybrid"}
    return config


def test_available_c_store_preserves_ordinary_predictions_and_independent_prior_calibration(
    mixed_seed, c_augmented_artifact
):
    enabled = RustForwardPassPerfModel.best_available(_augmented_config(mixed_seed, c_augmented_artifact))
    ordinary = RustForwardPassPerfModel.best_available(
        _augmented_config(mixed_seed, c_augmented_artifact, enabled=False)
    )
    missing = _load(mixed_seed, adaptation="frozen")
    try:
        query = _iteration()
        assert missing.estimate_forward_pass_time_ms(query) == 4.0  # P2 + D3 - D0(1).
        assert ordinary.estimate_forward_pass_time_ms(query) == pytest.approx(12.0)
        assert enabled.estimate_forward_pass_time_ms(query) == ordinary.estimate_forward_pass_time_ms(query)
        route = enabled.mixed_composition_route(query)
        assert route["fallback_eligible"] and not route["owned"]
        assert route["route"] == "ordinary"
        assert missing.mixed_composition_route(query)["owned"]
        assert enabled.mixed_composition_state() == missing.mixed_composition_state()

        initial = enabled.export_learned_artifact()
        for index in range(13):
            observation = _iteration(wall_ms=9 + index % 4, index=index)
            before = enabled.mixed_composition_state()
            prediction = enabled.estimate_forward_pass_time_ms(observation)
            assert prediction == ordinary.estimate_forward_pass_time_ms(observation)
            hidden_target = deepcopy(observation)
            hidden_target[0]["wall_time"] = 123.0
            assert enabled.estimate_forward_pass_time_ms(hidden_target) == prediction
            assert enabled.mixed_composition_state() == before
            assert enabled.estimate_forward_pass_detailed(observation)["latency_ms"] == prediction
            for model in (enabled, ordinary, missing):
                model.tune_with_fpms([observation])
            assert enabled.mixed_composition_state() == missing.mixed_composition_state()
            assert enabled.mixed_composition_route(query)["calibration_admissions"] == index + 1
            assert enabled.export_learned_artifact() == ordinary.export_learned_artifact() == initial
            assert enabled.estimate_forward_pass_time_ms(query) == ordinary.estimate_forward_pass_time_ms(query)

        # The ordinary correction is applied to the 12ms forest, while the
        # independently maintained interpolation correction is capped at 2x4ms.
        assert missing.estimate_forward_pass_time_ms(query) == 8.0
        assert enabled.estimate_forward_pass_time_ms(query) > 8.0
        snapshot = enabled.mixed_composition_state()
        pure = _iteration("decode", wall_ms=7)
        for model in (enabled, ordinary):
            model.tune_with_fpms([pure])
        assert enabled.mixed_composition_state() == snapshot
        assert enabled.estimate_forward_pass_time_ms(pure) == ordinary.estimate_forward_pass_time_ms(pure)
    finally:
        enabled.close()
        ordinary.close()
        missing.close()


def test_ready_c_store_mixed_snapshot_restores_and_rejects_old_precedence(mixed_seed, c_augmented_artifact, tmp_path):
    config = _augmented_config(mixed_seed, c_augmented_artifact)
    # Only mixed correction is exported by this API; keep ordinary correction
    # disabled so the resumed model's entire predictive state is accounted for.
    config["estimator_config"]["correction"]["enabled"] = False
    model = RustForwardPassPerfModel.best_available(config)
    restored = RustForwardPassPerfModel.best_available(config)
    try:
        for index in range(11):
            model.tune_with_fpms([_iteration(wall_ms=6 + index % 3, index=index)])
        snapshot = model.mixed_composition_state()
        restored.restore_mixed_composition_state(snapshot)
        assert restored.mixed_composition_state() == snapshot
        assert restored.mixed_composition_route(_iteration())["route"] == "ordinary"
        legacy = deepcopy(snapshot)
        legacy["version"] = "original_a_composition_attention_moe_v1"
        with pytest.raises(ValueError, match="saved mixed state does not match"):
            restored.restore_mixed_composition_state(legacy)
        assert restored.mixed_composition_state() == snapshot
        for index in range(11, 18):
            observation = _iteration(wall_ms=6 + index % 3, index=index)
            assert restored.estimate_forward_pass_time_ms(observation) == model.estimate_forward_pass_time_ms(
                observation
            )
            for current in (model, restored):
                current.tune_with_fpms([observation])
            assert restored.mixed_composition_state() == model.mixed_composition_state()

        changed = json.loads(c_augmented_artifact.read_text())
        changed["metadata"]["identity"]["model"] = "wrong/deployment"
        invalid = tmp_path / "wrong-c-identity.json"
        invalid.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="identity"):
            RustForwardPassPerfModel.best_available(_augmented_config(mixed_seed, invalid))
    finally:
        model.close()
        restored.close()


def test_mixed_checkpoint_preserves_pending_pure_rows_and_future_calibration(mixed_seed, tmp_path):
    model = _load(mixed_seed)
    resumed = None
    try:
        mixed = _iteration()
        assert model.estimate_forward_pass_time_ms(mixed) == 4.0  # P2 + D3 - D0(1).
        before = model.export_learned_artifact()
        assert model.mixed_composition_route(mixed)["owned"]
        assert model.mixed_composition_route(mixed)["calibration_admissions"] == 0
        for index in range(13):
            model.tune_with_fpms([_iteration(wall_ms=6 + index % 3, index=index)])
        assert model.export_learned_artifact() == before
        assert model.mixed_composition_route(mixed)["calibration_admissions"] == 13
        model.tune_with_fpms([_iteration("decode", wall_ms=7)])
        assert model.learned_online_diagnostics()["stores"]["pure_decode"]["pending_rows"] == 1
        checkpoint = tmp_path / "resume"
        state = model.mixed_composition_state()
        model.save_online_checkpoint(checkpoint)
        assert json.loads((checkpoint / "manifest.json").read_text())["schema_version"] == 5
        resumed = _load(mixed_seed, checkpoint)
        assert resumed.mixed_composition_state() == state
        assert resumed.export_learned_artifact() == model.export_learned_artifact()
        for index in range(13, 45):
            observation = _iteration("decode" if index % 3 == 0 else "mixed", wall_ms=5 + index % 5, index=index)
            assert resumed.estimate_forward_pass_time_ms(observation) == model.estimate_forward_pass_time_ms(
                observation
            )
            for current in (model, resumed):
                current.tune_with_fpms([observation])
            assert resumed.mixed_composition_state() == model.mixed_composition_state()
            assert resumed.export_learned_artifact() == model.export_learned_artifact()
        with pytest.raises(ValueError, match="policy is disabled"):
            _load(mixed_seed, checkpoint, enabled=False)
    finally:
        model.close()
        if resumed is not None:
            resumed.close()


def test_policy_off_schema5_continues_and_cannot_enable_on_runtime_checkpoint(mixed_seed, tmp_path):
    model = _load(mixed_seed, enabled=False)
    resumed = None
    try:
        assert model.mixed_composition_route(_iteration()) is None
        assert model.mixed_composition_diagnostics() is None
        assert model.mixed_composition_state() is None
        for index in range(7):
            model.tune_with_fpms([_iteration("decode", wall_ms=6, index=index)])
        checkpoint = tmp_path / "off-resume"
        model.save_online_checkpoint(checkpoint)
        assert json.loads((checkpoint / "manifest.json").read_text())["schema_version"] == 5
        resumed = _load(mixed_seed, checkpoint, enabled=False)
        assert resumed.export_learned_artifact() == model.export_learned_artifact()
        for index in range(7, 15):
            row = _iteration("decode", wall_ms=5 + index % 3, index=index)
            assert resumed.estimate_forward_pass_time_ms(row) == model.estimate_forward_pass_time_ms(row)
            for current in (model, resumed):
                current.tune_with_fpms([row])
            assert resumed.export_learned_artifact() == model.export_learned_artifact()
        with pytest.raises(ValueError, match="no mixed calibration state"):
            _load(mixed_seed, checkpoint)
    finally:
        model.close()
        if resumed is not None:
            resumed.close()


def test_public_snapshot_restore_is_atomic_and_selector_experts_stay_unchanged(mixed_seed, tmp_path):
    model = _load(mixed_seed, adaptation="causal_selector")
    restored = _load(mixed_seed, adaptation="causal_selector")
    try:
        experts = model.learned_expert_artifacts()
        selector = model.learned_selector_diagnostics()
        for index in range(11):
            model.tune_with_fpms([_iteration(wall_ms=6 + index % 3, index=index)])
        assert model.learned_expert_artifacts() == experts
        assert model.learned_selector_diagnostics() == selector
        checkpoint = tmp_path / "unsupported-selector-checkpoint"
        with pytest.raises(ValueError, match="selector"):
            model.save_online_checkpoint(checkpoint)
        assert not checkpoint.exists()
        snapshot = model.mixed_composition_state()
        restored.restore_mixed_composition_state(snapshot)
        assert restored.mixed_composition_state() == snapshot
        broken = deepcopy(snapshot)
        broken["admissions"]["contains_locally_mixed"] += 1
        with pytest.raises(ValueError, match="admission progress"):
            restored.restore_mixed_composition_state(broken)
        assert restored.mixed_composition_state() == snapshot
        for index in range(11, 40):
            row = _iteration(wall_ms=6 + index % 3, index=index)
            assert restored.estimate_forward_pass_time_ms(row) == model.estimate_forward_pass_time_ms(row)
            for current in (model, restored):
                current.tune_with_fpms([row])
            assert restored.mixed_composition_state() == model.mixed_composition_state()
    finally:
        model.close()
        restored.close()
