# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapter causality, source binding and native ownership (numerics are SDK-tested)."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from scripts.fpm_accuracy.models.learned_scenario import LearnedScenarioPredictor
from scripts.fpm_accuracy.models.mixed_scenario import MixedOriginalA
from scripts.fpm_accuracy.scenario_contract import GBDT_ONLINE_RECIPE
from scripts.fpm_accuracy.types.forward_pass import ForwardPassIteration, ForwardPassMetric, RequestMetrics


@dataclass
class Request:
    model: str = "Org/Model"
    system: str = "h200_sxm"
    worker_type: str = "aggregated"
    estimation_mode: str = "auto"
    fallback_policy: str = "deny"
    backend: str = "vllm"
    version: str = "0.25.1"
    tp_size: int = 4
    estimator_config: dict = field(default_factory=dict)


def observation(index, kind="decode", worker="worker", actual=10):
    return ForwardPassIteration.single_rank(
        ForwardPassMetric(
            fpm_id=index,
            configuration_id="config",
            version=1,
            worker_id=worker,
            dp_rank=0,
            counter_id=index,
            wall_time_s=actual / 1000,
            scheduled=RequestMetrics(
                num_prefill_requests=int(kind != "decode"),
                sum_prefill_tokens=32 * int(kind != "decode"),
                num_decode_requests=int(kind != "prefill"),
                sum_decode_kv_tokens=64 * int(kind != "prefill"),
            ),
            queued=RequestMetrics(),
            capture_id="trace",
        )
    )


def classify(rows, role):
    p = any(row["scheduled_requests"]["num_prefill_requests"] for row in rows)
    d = any(row["scheduled_requests"]["num_decode_requests"] for row in rows)
    return "contains_locally_mixed" if p and d else "pure_prefill" if p else "pure_decode" if d else None


class Native:
    models: ClassVar[list] = []

    @staticmethod
    def normalize_config(config):
        return deepcopy(config) if isinstance(config, dict) else asdict(config)

    @classmethod
    def best_available(cls, config):
        model = cls(config)
        cls.models.append(model)
        return model

    def __init__(self, config):
        self.config = config
        learned = config["estimator_config"]["fpm_learned"]
        path = Path(learned.get("artifact_path") or Path(learned["checkpoint_path"]) / "model.json")
        self.forest = json.loads(path.read_text())
        self.stable = deepcopy(self.forest)
        self.observations, self.appends, self.route_calls = [], [], 0
        self.closed = False

    def estimate_forward_pass_time_ms(self, rows):
        assert all("wall_time" not in row for row in rows)
        return 10 + len(self.observations)

    def estimate_forward_pass_with_route(self, rows):
        self.route_calls += 1
        return {
            "latency_ms": self.estimate_forward_pass_time_ms(rows),
            "temporary_mixed_composition": {"native_decision": self.route_calls},
        }

    def tune_with_fpms(self, rows):
        self.observations.extend(deepcopy(rows))

    def append_mixed_learned_checkpoint(self, path):
        artifact = json.loads((Path(path) / "model.json").read_text())
        assert not set(artifact["stores"]) & self.forest["stores"].keys()
        self.forest["stores"].update(artifact["stores"])
        self.appends.append({"rows_before_append": len(self.observations), "artifact": artifact})
        return 1

    def export_learned_artifact(self):
        return deepcopy(self.forest)

    def learned_expert_artifacts(self):
        return {"stable": deepcopy(self.stable), "online": deepcopy(self.forest)}

    def learned_online_diagnostics(self):
        return {"accepted": len(self.observations)}

    def learned_selector_diagnostics(self):
        return {"paired": len(self.observations)}

    def mixed_composition_diagnostics(self):
        return {"routes": [{"workload_kind": "contains_locally_mixed", "fallback_eligible": True}]}

    def diagnostics(self):
        return {"retained_observations": len(self.observations)}

    def close(self):
        self.closed = True


@pytest.fixture
def backend():
    import sklearn

    Native.models = []
    calls = []

    def train(rows, role, *, features, identity, online_checkpoint_dir=None, **settings):
        grouped = {}
        for row in rows:
            grouped.setdefault(classify(row, role), []).append(row)
        eligible = {kind: values for kind, values in grouped.items() if len(values) >= settings["min_store_rows"]}
        if not eligible:
            raise ValueError("every store fell below min_store_rows; nothing to export")
        calls.append({"rows": deepcopy(rows), "identity": deepcopy(identity), "settings": settings})
        artifact = {
            "stores": {
                kind: {"trees": [{"seed": len(values)}] * settings["max_iter"]} for kind, values in eligible.items()
            },
            "metadata": {"train_rows": {kind: len(values) for kind, values in eligible.items()}},
        }
        if online_checkpoint_dir is not None:
            online_checkpoint_dir.mkdir()
            (online_checkpoint_dir / "model.json").write_text(json.dumps(artifact))
            (online_checkpoint_dir / "manifest.json").write_text(json.dumps({"identity": identity}))
        return artifact

    trainer = SimpleNamespace(
        train=train,
        classify_workload=classify,
        FEATURE_PRESETS={"fpm5": GBDT_ONLINE_RECIPE["features"]},
        ONLINE_SUPPORTED_SKLEARN_VERSIONS={sklearn.__version__},
    )
    return trainer, calls


class Mixed:
    manifests: ClassVar[dict] = {}

    def configure_request(self, request):
        return request

    def bind(self, perf_model, trainer, config, rows, seed, minimum, directory, *, quality_gate):
        assert len(rows) == 40 and minimum == 20
        self.quality_gate = quality_gate
        config["estimator_config"]["temporary_mixed_composition"] = {"quality_gate": {}} if quality_gate else {}
        return config

    def close(self):
        pass


def make(backend, method="gbdt", rows=(), *, mixed=None, cold=False, online=True):
    return LearnedScenarioPredictor(
        method, Native, Request(), backend[0], rows, online=online, cold_start=cold, mixed=mixed
    )


@pytest.mark.parametrize("method", ["gbdt", "hybrid", "selector"])
def test_cold_start_uses_only_admitted_rows_and_keeps_workers_and_stores_separate(backend, method):
    model = make(backend, method, cold=method != "gbdt")
    try:
        for index in range(20):
            row = observation(index)
            assert model.predict(row.prediction_input()).value_ms is None
            model.tune([row])
        assert len(backend[1]) == 1
        assert len(backend[1][0]["rows"]) == 20
        native = Native.models[0]
        assert native.observations == []  # Seed labels are never replayed as feedback.
        row = observation(20)
        assert model.predict(row.prediction_input()).value_ms == 10
        assert model.predict(observation(20, worker="other").prediction_input()).value_ms is None
        assert model.predict(observation(20, kind="prefill").prediction_input()).value_ms is None
        model.tune([row])
        assert model.predict(row.prediction_input()).value_ms == 11
        assert backend[1][0]["identity"]["tp_size"] == 4
        assert backend[1][0]["identity"]["worker_type"] == "aggregated"
        assert backend[1][0]["settings"]["max_iter"] == 150
        assert len(model.seed_artifact_paths()) == 1
        for digest, path in model.seed_artifact_paths().items():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        model.diagnostics()
    finally:
        model.close()
    assert all(native.closed for native in Native.models)


@pytest.mark.parametrize("method", ["gbdt", "hybrid", "selector"])
def test_original_a_native_fallback_keeps_causal_append_and_frozen_hybrid(backend, method):
    original = [observation(i, kind) for kind in ("prefill", "decode") for i in range(20)]
    mixed = Mixed()
    model = make(backend, method, original, mixed=mixed)
    try:
        native = Native.models[0]
        assert mixed.quality_gate is (method == "gbdt")
        for index in range(21):
            row = observation(index, "mixed")
            prediction = model.predict(row.prediction_input())
            assert prediction.metadata == {"temporary_mixed_composition": {"native_decision": index + 1}}
            if index < 20:
                assert native.appends == []
            model.tune([row])
        if method == "hybrid":
            assert len(backend[1]) == 1 and native.appends == []
            assert "contains_locally_mixed" not in native.forest["stores"]
        else:
            assert len(backend[1]) == 2 and len(native.appends) == 1
            assert native.appends[0]["rows_before_append"] == 20
            assert len(backend[1][1]["rows"]) == 20
        assert len(native.observations) == 21
        model.diagnostics()
    finally:
        model.close()


def test_frozen_test_labels_are_rejected(backend):
    model = make(backend, rows=[observation(i) for i in range(20)], online=False)
    try:
        with pytest.raises(ValueError, match="cannot admit"):
            model.tune([observation(21)])
        assert Native.models[0].observations == []
    finally:
        model.close()


def test_below_threshold_b_store_only_finishes_after_admitted_c(backend):
    model = make(backend, rows=[observation(i) for i in range(18)])
    try:
        assert model.predict(observation(18).prediction_input()).value_ms is None
        model.tune([observation(18)])
        assert not backend[1]
        model.tune([observation(19)])
        assert len(backend[1][0]["rows"]) == 20
        assert not Native.models[0].observations
    finally:
        model.close()


def test_original_a_binding_uses_native_quality_defaults_and_exact_sources(tmp_path):
    mixed = object.__new__(MixedOriginalA)
    mixed.scenario, mixed.manifests, mixed.provenance = "ac", {}, {"sources": [{"sha256": "a" * 64}]}
    seen = {}

    def write(path, rows, worker_type, **kwargs):
        seen.update(kwargs)
        return {"path": str(path), "source_sha256": "b" * 64}

    config = {"worker_type": "aggregated", "estimator_config": {}}
    authored = mixed.bind(
        Native,
        SimpleNamespace(write_original_a_manifest=write),
        config,
        [[{}]],
        tmp_path / "seed.json",
        20,
        tmp_path,
        quality_gate=True,
    )
    policy = authored["estimator_config"]["temporary_mixed_composition"]
    assert policy["quality_gate"] == {}  # Rust fills defaults; adapter must not replace them.
    assert policy["scenario"] == "a_c" and policy["mode"] == "compose_calibrated"
    assert seen["provenance"] == mixed.provenance
    assert seen["seed_artifact_paths"] == [tmp_path / "seed.json"]
    assert config["estimator_config"] == {}


def test_failed_native_admission_does_not_seed_a_missing_mixed_store(backend):
    original = [observation(i, kind) for kind in ("prefill", "decode") for i in range(20)]
    model = make(backend, rows=original, mixed=Mixed())
    try:

        def reject(_):
            raise ValueError("invalid measured metrics")

        Native.models[0].tune_with_fpms = reject
        with pytest.raises(ValueError, match="invalid measured"):
            model.tune([observation(100, "mixed")])
        assert model.diagnostics()["scopes"][0]["pending_bootstrap"] == {}
        assert len(backend[1]) == 1
    finally:
        model.close()


def test_untrainable_bootstrap_remains_bounded(backend):
    model = make(backend)
    try:
        model.tune([observation(i) for i in range(19)])

        def reject(*args, **kwargs):
            raise ValueError("every store fell below min_store_rows; nothing to export")

        backend[0].train = reject
        for index in range(19, 23):
            with pytest.raises(ValueError, match="did not produce a trainable"):
                model.tune([observation(index)])
            assert model.diagnostics()["scopes"][0]["pending_bootstrap"] == {"pure_decode": 19}
    finally:
        model.close()


def test_published_native_evidence_keeps_hashes_and_removes_local_paths(backend):
    model = make(backend, rows=[observation(i) for i in range(20)])
    try:
        Native.models[0].diagnostics = lambda: {
            "source": "/private/tmp/seed/model.json",
            "source_sha256": "a" * 64,
            "config": {"systems_paths": ["/home/cache/tables"], "model": "Org/Model"},
        }
        native = model.diagnostics()["scopes"][0]["native"]
        assert native["source"] == "[local path omitted]"
        assert native["source_sha256"] == "a" * 64
        assert native["config"]["systems_paths"] == ["[local path omitted]"]
        assert native["config"]["model"] == "Org/Model"
    finally:
        model.close()


def test_bounded_diagnostics_keep_complete_config_fingerprint_and_native_evidence():
    from scripts.fpm_accuracy.models.learned_scenario import _bounded_native_diagnostics, _digest

    config = {
        "model": "Org/Model",
        "system": "h200_sxm",
        "backend": "vllm",
        "worker_type": "aggregated",
        "tp": 4,
        "gemm_quant_mode": "fp8",
        "systems_paths": ["/private/tmp/native-tables"],
        "estimator_config": {"fpm_learned": {"adaptation": "online", "checkpoint_path": "/private/tmp/seed"}},
        "fpm_profile": {"model": "Org/Model", "deployments": [{"tp": 4}]},
    }
    source = {
        "provenance": {"config": config, "selected_estimation_mode": "fpm_learned"},
        "stores": {"decode": {"retained_row_ids": list(range(10000)), "retained_row_ids_sha256": "a" * 64}},
        "mixed": {"policy": {"quality_gate": {"alpha": 0.05}}, "paired_comparisons": 35, "source_sha256": "b" * 64},
    }
    result = _bounded_native_diagnostics(source)
    provenance = result["provenance"]
    assert provenance["config_sha256"] == _digest(config)
    assert provenance["estimator_config_sha256"] == _digest(config["estimator_config"])
    assert provenance["fpm_profile_sha256"] == _digest(config["fpm_profile"])
    assert provenance["deployment_identity"]["tp"] == 4
    assert provenance["deployment_identity"]["gemm_quant_mode"] == "fp8"
    assert provenance["selected_estimation_mode"] == "fpm_learned"
    assert result["stores"]["decode"] == {"retained_row_count": 10000, "retained_row_ids_sha256": "a" * 64}
    assert result["mixed"] == source["mixed"]
    assert len(json.dumps(result)) < 2000
    assert "/private/tmp" not in json.dumps(result)
    assert source["provenance"]["config"] == config


def test_old_native_schema_is_unsupported_but_invalid_input_remains_error():
    from scripts.fpm_accuracy.exceptions import DependencyError
    from scripts.fpm_accuracy.models.learned_scenario import _canonical_normalize

    class Old:
        @staticmethod
        def normalize_config(_):
            raise ValueError("unknown field `adaptation`")

    with pytest.raises(DependencyError, match="lacks"):
        _canonical_normalize(Old, {})

    class Invalid:
        @staticmethod
        def normalize_config(_):
            raise ValueError("tp must be positive")

    with pytest.raises(ValueError, match="tp must"):
        _canonical_normalize(Invalid, {})


def test_native_default_sections_may_be_omitted_from_normalized_configuration(backend):
    class DefaultsNative(Native):
        @staticmethod
        def normalize_config(config):
            normalized = Native.normalize_config(config)
            estimator = normalized["estimator_config"]
            learned = estimator.get("fpm_learned", {})
            if learned.get("adaptation") == "online" and not learned.get("checkpoint_path"):
                estimator.pop("fpm_learned", None)
            estimator.pop("selector", None)
            return normalized

    for method in ("gbdt", "selector"):
        model = LearnedScenarioPredictor(
            method, DefaultsNative, Request(), backend[0], online=True, cold_start=method == "selector"
        )
        model.close()
